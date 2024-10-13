% psk-16  理论的错误率，高斯白噪声信道下
%%%%%%% generate data 
clear 

M = 16;
N = 10000000;       %这里的次数如果太少，则在信噪比好的情况下，错误数是 0. 样本数量不够。大约到 10000000
data = randi([0,M-1],[1,N]);

% modulated
txData = pskmod(data,M);

% add noise
EsN0 = (0:27);             % symbol energy / (Noise energy per HZ)，因为一个赫兹就一个符号，所以，也就是 SNR.
bers = zeros(size(EsN0));

for i = (1:length(EsN0))
    esn0 = EsN0(i);
    Pnoise = 1/(10^(esn0/10));            % 每个符号的能量是 1，即实部的平方 + 虚部的平方 = 1
    noiseData = sqrt(Pnoise/2)*randn(1,N) + sqrt(Pnoise/2)*randn(1,N) * 1i;     %噪声的实部能量占 1/2，虚部占 1/2   
    rxData = txData + noiseData;

    %% de-modulation
    recData = pskdemod(rxData,M);

    Nerror = sum( not (data == recData));

    bers(i) = Nerror/(N*1.0);
    
end

%% 理论上的错误率
theorySer_16PSK = erfc(sqrt(10.^(EsN0/10))*sin(pi/M));


figure
semilogy(EsN0,bers,'bs-','LineWidth',2);
hold on
semilogy(EsN0,theorySer_16PSK,'mx-','LineWidth',2);

axis([0 25 10^-5 1])
grid on
legend('simulated','theory-16PSK');
xlabel('Es/No, dB')
ylabel('Symbol Error Rate')
title('Symbol error probability curve for 16-PSK')
