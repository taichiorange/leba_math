clear;
Nts = [4,8,16];
Nrs = [4,8,16];

N = 1e5;

SNRs = (-10:1:20);

BERs_MF = zeros(length(Nts), length(SNRs));
BERs_ZF = zeros(length(Nts), length(SNRs));
BERs_LMMSE = zeros(length(Nts), length(SNRs));

tic
for cc=(1:length(Nts))
    Nt = Nts(cc);
    Nr = Nrs(cc);
    
    for ee=(1:length(SNRs))
        SNR = SNRs(ee);
        Pnoise = 1/10^(SNR/10);
        NerrorMF = 0;
        NerrorZF = 0;
        NerrorLMMSE = 0;
        
        for ii=(1:N)
            x = 2*randi([0,1],Nt,1) - 1;
            H = 1/sqrt(Nt) * (randn(Nr,Nt) + 1j*randn(Nr,Nt))/sqrt(2);
            n = sqrt(Pnoise/2) * (randn(Nr,1) + 1j * randn(Nr,1));
            
            y = H*x + n;
            
            % matched filter
            x_hat = H'* y;
            x_hat = 2*(x_hat>0)-1;
            NerrorMF = NerrorMF + sum(x ~= x_hat);
            
            % zero forcing
            x_hat = (H'* H) \ H' * y;
            x_hat = 2*(x_hat>0)-1;
            NerrorZF = NerrorZF + sum( x ~= x_hat);
            
            % LMMSE
            x_hat = (H'*H + Pnoise * diag(ones(Nt,1)))\H' * y;
            x_hat = 2*(x_hat>0)-1;
            NerrorLMMSE = NerrorLMMSE + sum( x ~= x_hat);
        end
        
        BERs_MF(cc,ee) = NerrorMF;
        BERs_ZF(cc,ee) = NerrorZF;
        BERs_LMMSE(cc,ee) = NerrorLMMSE;
    end

    BERs_MF(cc,:) = BERs_MF(cc,:) /(Nt* N* 1.0);
    BERs_ZF(cc,:) = BERs_ZF(cc,:) /(Nt* N* 1.0);
    BERs_LMMSE(cc,:) = BERs_LMMSE(cc,:) /(Nt* N* 1.0);

end

toc;



figure();
semilogy(SNRs, BERs_MF(1,:), "k*--");
hold on;
semilogy(SNRs, BERs_MF(2,:), "b>--");
hold on;
semilogy(SNRs, BERs_MF(3,:), "rd--");
hold on;
% save('MIMO_detection_MF_4x4_8x8_16x16.mat','SNRs');
% save('MIMO_detection_MF_4x4_8x8_16x16.mat','BERs_MF','-append');

semilogy(SNRs, BERs_ZF(1,:), "k*-");
hold on;
semilogy(SNRs, BERs_ZF(2,:), "b>-");
hold on;
semilogy(SNRs, BERs_ZF(3,:), "rd-");
hold on;

% save('MIMO_detection_ZF_4x4_8x8_16x16.mat','SNRs');
% save('MIMO_detection_ZF_4x4_8x8_16x16.mat','BERs_ZF','-append');


semilogy(SNRs, BERs_LMMSE(1,:), "k*:");
hold on;
semilogy(SNRs, BERs_LMMSE(2,:), "b>:");
hold on;
semilogy(SNRs, BERs_LMMSE(3,:), "rd:");
hold on;

% save('MIMO_detection_LMMSE_4x4_8x8_16x16.mat','SNRs');
% save('MIMO_detection_LMMSE_4x4_8x8_16x16.mat','BERs_LMMSE','-append');

legend("4MF", "8MF", "16MF", "4ZF", "8ZF", "16ZF", "4LMMSE", "8LMMSE","16LMMSE");
ylim([0.00001,1]);
grid on;