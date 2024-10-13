% 16 QAM  理论的错误率，高斯白噪声信道下
%%%%%%% generate data 
clear 

M = 16;
N = 10000000;       %这里的次数如果太少，则在信噪比好的情况下，错误数是 0. 样本数量不够。10000000 的时候，在 23dB 时，就已经错误率为 0 了 
data = randi([0,M-1],[1,N]);

% modulated
txData = qammod(data,M);

% add noise
EsN0 = (0:27);             % symbol energy / (Noise energy per HZ)，因为一个赫兹就一个符号，所以，也就是 SNR.
bers = zeros(size(EsN0));

for i = (1:length(EsN0))
    esn0 = EsN0(i);
    Pnoise = 10/(10^(esn0/10));            % 每个符号的平均能量是 10，即 实部的平方 + 虚部的平方， 16 种情况取平均
    noiseData = sqrt(Pnoise/2)*randn(1,N) + sqrt(Pnoise/2)*randn(1,N) * 1i;     %噪声的实部能量占 1/2，虚部占 1/2   
    rxData = txData + noiseData;

    %% de-modulation
    recData = qamdemod(rxData,M);

    Nerror = sum( not (data == recData));

    bers(i) = Nerror/(N*1.0);
    
end

%% 理论上的错误率
theorySer_16QAM = 3/2*erfc(sqrt(0.1*(10.^(EsN0/10))));

figure
semilogy(EsN0,bers,'bs-','LineWidth',2);
hold on
semilogy(EsN0,theorySer_16QAM,'mx-','LineWidth',2);

axis([0 25 10^-10 1])
grid on
legend('simulated','theory-16QAM');
xlabel('Es/No, dB')
ylabel('Symbol Error Rate')
title('Symbol error probability curve for 16-16QAM')
