import os
import time

if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print("no of GPUs:", gpus)
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

import numpy as np
import sionna as sn
import matplotlib.pyplot as plt

# 显式导入 Sionna 组件
from sionna.phy.mapping import Constellation, Mapper, Demapper, BinarySource
from sionna.phy.utils import  ebnodb2no, insert_dims, log10, expand_to_rank
from sionna.phy.channel import AWGN
from tensorflow.keras.layers import Layer, Conv1D, LayerNormalization
from tensorflow.keras.activations import relu
from sionna.phy import Block

# ==========================================
# 0. 全局参数配置
# ==========================================
NUM_BITS_PER_SYMBOL = 6  # 16QAM (可配: 2=QPSK, 4=16QAM, 6=64QAM)
BLOCK_LENGTH = 123      # 每帧传输的符号数量 (时域序列长度)
BATCH_SIZE = 199         # 训练时的 Batch Size
NUM_CONV_CHANNELS = 268  # 卷积层的通道数/特征数
KERNEL_SIZE = 5         # 卷积核大小

LEARNING_RATE = 1e-4     # 学习率
TRAINING_STEPS = 6000    # 训练迭代次数
EBN0_DB_MIN = 0        # 训练信噪比最小值
EBN0_DB_MAX = 20.0       # 训练信噪比最大值


# ==========================================
# 1. 神经网络组件 (改为 1D)
# ==========================================

class ResidualBlock1D(Layer):
    """
    一维残差块: 用于处理时域序列
    结构: Norm -> ReLU -> Conv1D -> Norm -> ReLU -> Conv1D -> Add
    """
    def build(self, input_shape):
        # 1D 归一化: axis=(-1, -2) 对应 (Features, Time)
        self._layer_norm_1 = LayerNormalization(axis=(-1, -2))
        self._conv_1 = Conv1D(filters=NUM_CONV_CHANNELS,
                              kernel_size=KERNEL_SIZE,
                              padding='same',
                              activation=None)
        
        self._layer_norm_2 = LayerNormalization(axis=(-1, -2))
        self._conv_2 = Conv1D(filters=NUM_CONV_CHANNELS,
                              kernel_size=KERNEL_SIZE,
                              padding='same',
                              activation=None)

    def call(self, inputs):
        z = self._layer_norm_1(inputs)
        z = relu(z)
        z = self._conv_1(z)
        z = self._layer_norm_2(z)
        z = relu(z)
        z = self._conv_2(z)
        # Skip connection
        z = z + inputs
        return z

class NeuralReceiver1D(Layer):
    """
    一维神经网络接收机
    输入: 接收信号 y (复数), 噪声功率 no
    输出: LLRs (Log-Likelihood Ratios)
    """
    def build(self, input_shape):
        # 输入层卷积
        self._input_conv = Conv1D(filters=NUM_CONV_CHANNELS,
                                  kernel_size=KERNEL_SIZE,
                                  padding='same',
                                  activation=None)
        
        # 堆叠 4 个残差块 (可以根据需要增加深度)
        self._res_blocks = [ResidualBlock1D() for _ in range(1)]
        
        # 输出层: 输出通道数 = 每个符号的比特数 (对应 LLR)
        self._output_conv = Conv1D(filters=NUM_BITS_PER_SYMBOL,
                                   kernel_size=KERNEL_SIZE,
                                   padding='same',
                                   activation=None)

    def call(self, y, no):
        # y: [batch, time] (复数)
        # no: [batch] (标量)

        # 1. 数据预处理
        # 将噪声功率转为对数域
        no = log10(no) 

        # 扩展维度以便拼接
        # y_real/imag: [batch, time, 1]
        y_real = tf.expand_dims(tf.math.real(y), axis=-1)
        y_imag = tf.expand_dims(tf.math.imag(y), axis=-1)
        
        # no 处理: [batch] -> [batch, 1, 1] -> [batch, time, 1]
        no = expand_to_rank(no, 3, axis=1)
        no = tf.tile(no, [1, tf.shape(y)[1], 1])

        # 拼接: [batch, time, 3] (Real, Imag, Noise)
        z = tf.concat([y_real, y_imag, no], axis=-1)

        # 2. 神经网络前向传播
        z = self._input_conv(z)
        
        for block in self._res_blocks:
            z = block(z)
            
        # 输出 LLR Logits: [batch, time, num_bits]
        logits = self._output_conv(z)
        
        return logits

# ==========================================
# 2. 端到端系统模型 (包含 Baseline 和 Neural Rx)
# ==========================================

class E2ESystem(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        # --- 通用组件 ---
        # 调制方式: "qam", num_bits
        self.constellation = Constellation("qam", NUM_BITS_PER_SYMBOL)
        self.mapper = Mapper(constellation=self.constellation)
        self.binary_source = BinarySource()
        self.awgn_channel = AWGN()
        
        # --- 方案 A: 神经网络接收机 ---
        self.neural_receiver = NeuralReceiver1D()
        
        # --- 方案 B: 传统最优接收机 (Baseline) ---
        # "app" = A Posteriori Probability (最优解)
        #self.baseline_demapper = Demapper("app", constellation=self.constellation)
        self.baseline_demapper = Demapper("maxlog", constellation=self.constellation)
        
        # Loss 函数 (用于训练神经网络)
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, batch_size, ebn0_db):
        """
        前向传播逻辑
        """
        # 1. 发射机
        # 生成比特: [batch, block_length * num_bits]
        num_bits_total = batch_size * BLOCK_LENGTH * NUM_BITS_PER_SYMBOL
        b = self.binary_source([batch_size, BLOCK_LENGTH * NUM_BITS_PER_SYMBOL])
        
        # 调制: [batch, block_length] (复数符号)
        x = self.mapper(b)
        
        # 2. 信道 (AWGN)
        # 将 Eb/N0 转换为噪声方差 No
        no = ebnodb2no(ebn0_db, NUM_BITS_PER_SYMBOL, coderate=1.0)
        
        # 加噪
        y = self.awgn_channel(x, no)
        
        # 3. 接收机处理
        
        # --- 路径 A: 神经网络 ---
        # 输出形状: [batch, block_length, num_bits]
        llr_neural = self.neural_receiver(y, no)
        # 展平以便计算 Loss/BER: [batch, total_bits]
        llr_neural_flat = tf.reshape(llr_neural, [batch_size, -1])

        # --- 路径 B: Baseline (最优解) ---
        # 输入需要是 [batch, block_length]
        # 输出形状: [batch, block_length * num_bits]
        no_eff_= expand_to_rank(no, tf.rank(y))
        llr_baseline = self.baseline_demapper(y, no_eff_)
       
        return b, llr_neural_flat, llr_baseline

# ==========================================
# 3. 训练与验证流程
# ==========================================

def run_training_and_eval():

    # 记录开始时间
    start_time = time.time()
    
    # 初始化模型
    model = E2ESystem()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # --- 训练循环 ---
    print(f"开始训练 (16QAM, AWGN)... 总步数: {TRAINING_STEPS}")

    # 这个是用于生成静态图，可以用 GPU 并行运算来加速，但是，不方便单步调试。
    # 如果要单步调试，可以把 @tf.function 注释掉
    @tf.function(jit_compile=True) 
    def train_step(batch_size):
        # 训练时随机采样 Eb/N0，增加鲁棒性
        ebn0_db = tf.random.uniform([batch_size], EBN0_DB_MIN, EBN0_DB_MAX)
        
        with tf.GradientTape() as tape:
            # 前向传播
            b, llr_neural, _ = model(batch_size=batch_size, ebn0_db=ebn0_db)
            # 计算 Loss
            loss = model.bce_loss(b, llr_neural)
            
        # 反向传播更新权重
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return loss

    for step in range(TRAINING_STEPS):
        # 调用静态图函数
        # 注意：这里传的是 Tensor 还是 Python int 都可以，
        # 但如果是 Python int，TF 可能会针对每个不同的值重新 Trace 一次图。
        # 既然 BATCH_SIZE 是固定的，这没问题。
        loss = train_step(BATCH_SIZE)
        if step % 500 == 0:
            print(f"Step {step}: Loss = {loss.numpy():.4f}")

    print(f"Step {step}: Loss = {loss.numpy():.4f}")        
    print("训练完成。\n")

    # --- 性能评估与对比 ---
    print("开始性能评估 (对比 Neural Rx vs Optimal APP)...")
    
    # 测试点: 0dB 到 14dB
    ebn0_range = np.arange(0, 20, 1.0)
    ber_neural = []
    ber_baseline = []
    
    # 为了测试准确，使用更大的 Batch Size
    test_batch_size = 1000 
    
    for ebn0 in ebn0_range:
        # 固定 Eb/N0
        ebn0_tensor = tf.fill([test_batch_size], float(ebn0))
        
        # 运行模型
        b, llr_neural, llr_baseline = model(batch_size=test_batch_size, ebn0_db=ebn0_tensor)
        
        # 计算 Neural Rx 的误码率
        # 硬判决: Logits > 0 判为 1 (视具体映射而定，这里只要两边一致即可)
        b_neural = sn.phy.utils.hard_decisions(llr_neural)
        ber_nn = sn.phy.utils.compute_ber(b, b_neural).numpy()
        ber_neural.append(ber_nn)
        
        # 计算 Baseline 的误码率
        # 硬判决: Logits > 0 判为 1 (视具体映射而定，这里只要两边一致即可)
        b_base = sn.phy.utils.hard_decisions(llr_baseline)
        ber_bl = sn.phy.utils.compute_ber(b, b_base).numpy()
        ber_baseline.append(ber_bl)
        
        print(f"Eb/N0 = {ebn0} dB | Neural BER: {ber_nn:.6f} | Optimal BER: {ber_bl:.6f}")
        
    # 记录结束时间
    end_time = time.time()
    # 计算运行时间
    run_time = end_time - start_time
    print(f"代码运行时间：{run_time:.6f} 秒")  # 保留6位小数
    
    # --- 绘图 ---
    try:
        plt.figure(figsize=(8, 6))
        plt.semilogy(ebn0_range, ber_baseline, 'k--o', label='Optimal App Demapper')
        plt.semilogy(ebn0_range, ber_neural, 'r-x', label='Neural Receiver')
        plt.grid(True, which="both", ls="-")
        plt.xlabel('Eb/N0 (dB)')
        plt.ylabel('Bit Error Rate (BER)')
        plt.title(f'BER Performance: 16QAM AWGN (Neural vs Optimal)')
        plt.legend()
        plt.ylim(1e-5, 1)
        plt.show()
    except Exception as e:
        print("绘图失败 (可能是环境原因):", e)

# 运行主函数
if __name__ == "__main__":
    run_training_and_eval()

