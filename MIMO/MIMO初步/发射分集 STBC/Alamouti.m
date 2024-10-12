clear;
close all;

SNRs = (0:1:10);
N = 1e5;
BERs_Alamouti = zeros(size(SNRs));
nErr =  zeros(size(SNRs));
tic
for eee=(1:length(SNRs))
    snr = SNRs(eee);
    Pnoise = 1/10^(snr/10);
    
    Datas = randi([0,1],1,N);
    Dmod = pskmod(Datas,2)/sqrt(2) ;     
    
    H = (randn(2,N)+ 1j * randn(2,N))/sqrt(2);
    
    % for Alamouti
    h = 1/sqrt(2)*(randn(1,N) + 1j*randn(1,N)); % Rayleigh channel
    %HH = (randn(2,N/2) + 1j * randn(2,N/2) )/sqrt(2);
    HH = reshape(h,2,N/2);
    
    Noise_Alamouti = sqrt(Pnoise/2)*(randn(1,N) + 1j * randn(1,N));
    %Noise_Alamouti = Noise_Alamouti+sqrt(Pnoise/2)*(randn(1,N) + 1j * randn(1,N));
    %Noise_Alamouti = Noise_Alamouti+sqrt(Pnoise/2)*(randn(1,N) + 1j * randn(1,N));
    %Noise_Alamouti = 1/sqrt(2)*[randn(1,N) + 1j*randn(1,N)];
    
    DmodAlam = zeros(1,N);
    for ii = (1:N)
        if(mod(ii,2)==1)
            DmodAlam(ii) = Dmod(ii) * HH(1,(ii+1)/2);
            DmodAlam(ii) = Dmod(ii+1) * HH(2,(ii+1)/2) + DmodAlam(ii);
        else
            DmodAlam(ii) = -conj(Dmod(ii)) * HH(1,ii/2);
            DmodAlam(ii) = conj(Dmod(ii-1)) * HH(2,ii/2) + DmodAlam(ii);
        end
    end
    
    R_eq = DmodAlam +  Noise_Alamouti;
    S = zeros(1,N);
    for ii=(1:N)
        if(mod(ii,2)==1)
            S(ii) = HH(1,(ii+1)/2)' * R_eq(ii);
            S(ii) = HH(2,(ii+1)/2) * (R_eq(ii+1)') + S(ii);
            LHH = HH(1,(ii+1)/2)' * HH(1,(ii+1)/2) + HH(2,(ii+1)/2)' * HH(2,(ii+1)/2);
            S(ii) = S(ii)/ LHH;
        else
            S(ii) = HH(2,ii/2)' * R_eq(ii-1);
            S(ii) = -HH(1,ii/2) * (R_eq(ii)') + S(ii);
            LHH = HH(1,ii/2)' * HH(1,ii/2) + HH(2,ii/2)' * HH(2,ii/2);
            S(ii) = S(ii)/ LHH;
        end
    end
    Datas_rcv = pskdemod(S,2);
    BERs_Alamouti(eee) = sum(not(Datas_rcv == Datas));    
    
    
end
toc

EbN0Lin = 10.^(SNRs/10);
pAlamouti = 1/2 - 1/2*(1+2./EbN0Lin).^(-1/2);
theoryBerAlamouti_nTx2_nRx1 = pAlamouti.^2.*(1+2*(1-pAlamouti)); 

figure(1);
semilogy(SNRs, BERs_Alamouti/(N*1.0),"rd-");
hold on;
semilogy(SNRs,theoryBerAlamouti_nTx2_nRx1, "g*-");


legend("Alamouti","theory 21 Alamouti");
grid on;