clear;
Nts = [1,2,4,8,16];
Nrs = [1,2,4,8,16];

N = 1e5;

SNRs = (-10:1:30);

BERs_MF = zeros(length(Nts), length(SNRs));

tic
for cc=(1:length(Nts))
    Nt = Nts(cc);
    Nr = Nrs(cc);

    parfor ee=(1:length(SNRs))
        SNR = SNRs(ee);
        Pnoise = 1/10^(SNR/10);
        NerrorMF = 0;

        for ii=(1:N)
            x = 2*randi([0,1],Nt,1) - 1;
            H = 1/sqrt(Nt) * (randn(Nr,Nt) + 1j*randn(Nr,Nt))/sqrt(2);
            n = sqrt(Pnoise/2) * (randn(Nr,1) + 1j * randn(Nr,1));

            y = H*x + n;

            % matched filter
            x_hat = H'* y;
            x_hat = 2*(x_hat>0)-1;
            NerrorMF = NerrorMF + sum(x ~= x_hat);

        end

        BERs_MF(cc,ee) = NerrorMF;
    end

    BERs_MF(cc,:) = BERs_MF(cc,:) /(Nt* N* 1.0);

end

toc;



figure();
semilogy(SNRs(1:25), BERs_MF(1,(1:25)), "k*--");
hold on;
semilogy(SNRs, BERs_MF(2,:), "b>--");
hold on;
semilogy(SNRs, BERs_MF(3,:), "rd--");
hold on;
semilogy(SNRs, BERs_MF(4,:), "gd--");
hold on;
semilogy(SNRs, BERs_MF(5,:), "k-");
hold on;
legend("1x1","2x2","4x4","8x8","16x16");
grid on; 


            
            
            
            
            
            