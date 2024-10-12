clear;
Nt = 8;
Nr = Nt;

N = 1e4;

SNRs = (-10:2:30);

BERs_ZF = zeros(size(SNRs));
BERs_ML = zeros(size(SNRs));


CC = 0:1:2^Nt-1;
CC = dec2bin(CC)-'0';
CC = CC';

XX = 2*CC - 1;

for ee=(1:length(SNRs))
    SNR = SNRs(ee);
    Pnoise = 1/10^(SNR/10);

    NerrorZF = 0;
    NerrorML = 0;
    
    for ii=(1:N)
        c = randi([0,1],Nt,1);
        x = 2*c - 1;
        H = 1/sqrt(Nt) * (randn(Nr,Nt) + 1j*randn(Nr,Nt))/sqrt(2);
        n = sqrt(Pnoise/2) * (randn(Nr,1) + 1j * randn(Nr,1));
        
        y = H*x + n;
        
        % zero forcing
        x_hat = (H'* H) \ H' * y;
        x_hat = 2*(x_hat>0)-1;
        NerrorZF = NerrorZF + sum( x ~= x_hat);
    
        % Maximum Likelihood
        [val,index]=min(sum(abs(y - H*XX).^2,1));
        c_hat = CC(:,index);
    
        NerrorML = NerrorML + sum(c_hat ~= c,'all');
    end
    
    BERs_ZF(ee) = NerrorZF;
    BERs_ML(ee) = NerrorML;
end

BERs_ZF = BERs_ZF /(Nt* N* 1.0);
BERs_ML = BERs_ML /(Nt* N* 1.0);


%figure();
semilogy(SNRs, BERs_ZF, "k*-.");
hold on;
semilogy(SNRs, BERs_ML, "R>-.");

legend("ZF", "ML");
grid on;