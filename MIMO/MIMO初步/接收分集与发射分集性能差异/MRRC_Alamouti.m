clear;
close all;

SNRs = (0:0.3:5);
N = 1e5;
BERs_MRRC = zeros(size(SNRs));
BERs_Alamouti = zeros(size(SNRs));
nErr =  zeros(size(SNRs));
tic
for eee=(1:length(SNRs))
    snr = SNRs(eee);
    Pnoise = 1/10^(snr/10);
    
    Datas = randi([0,1],1,N);
    Dmod = pskmod(Datas,2) ;
    
    H = (randn(2,N)+ 1j * randn(2,N))/sqrt(2);
    
    % for MRRC  tx = 1  rx = 2
    Noise_MRRC = sqrt(Pnoise/2)*(randn(2,N) + 1j * randn(2,N));
    DmodMRRC = zeros(2,N);
    DmodMRRC(1,:) = Dmod;
    DmodMRRC(2,:) = Dmod;
    Rmrrc = H .* DmodMRRC + Noise_MRRC;
    R_eq = sum(Rmrrc .* conj(H),1) ./ sum(conj(H).*H,1);    % transmiting
    Datas_rcv = pskdemod(R_eq,2);
    BERs_MRRC(eee) = sum(not(Datas_rcv == Datas));
    
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
            DmodAlam(ii) = Dmod(ii)/sqrt(2) * HH(1,(ii+1)/2);
            DmodAlam(ii) = Dmod(ii+1)/sqrt(2) * HH(2,(ii+1)/2) + DmodAlam(ii);
        else
            DmodAlam(ii) = -conj(Dmod(ii)/sqrt(2)) * HH(1,ii/2);
            DmodAlam(ii) = conj(Dmod(ii-1)/sqrt(2)) * HH(2,ii/2) + DmodAlam(ii);
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
    
    
    %===========================================
    % from dsp log
    % Transmitter
    ip = rand(1,N)>0.5; % generating 0,1 with equal probability
    s = 2*ip-1; % BPSK modulation 0 -> -1; 1 -> 0

    % Alamouti STBC 
    sCode = zeros(2,N);
    sCode(:,1:2:end) = (1/sqrt(1))*reshape(s,2,N/2); % [x1 x2  ...]
    sCode(:,2:2:end) = (1/sqrt(1))*(kron(ones(1,N/2),[-1;1]).*flipud(reshape(conj(s),2,N/2))); % [-x2* x1* ....]

    %h = 1/sqrt(2)*[randn(1,N) + j*randn(1,N)]; % Rayleigh channel
    hMod = kron(reshape(h,2,N/2),ones(1,2)); % repeating the same channel for two symbols    

    %n = 1/sqrt(2)*(randn(1,N) + 1i*randn(1,N)); % white gaussian noise, 0dB variance

    % Channel and noise Noise addition
    %y = sum(hMod.*sCode,1) + 10^(-snr/20)*n;
    %y = sum(hMod.*sCode,1) + 10^(-snr/20)*Noise_Alamouti;
    y = sum(hMod.*sCode/sqrt(2),1) + Noise_Alamouti;
    %y = sum(hMod.*sCode,1) + Noise_Alamouti;

    % Receiver
    yMod = kron(reshape(y,2,N/2),ones(1,2)); % [y1 y1 ... ; y2 y2 ...]
    yMod(2,:) = conj(yMod(2,:)); % [y1 y1 ... ; y2* y2*...]
 
    % forming the equalization matrix
    hEq = zeros(2,N);
    hEq(:,(1:2:end)) = reshape(h,2,N/2); % [h1 0 ... ; h2 0...]
    hEq(:,(2:2:end)) = kron(ones(1,N/2),[1;-1]).*flipud(reshape(h,2,N/2)); % [h1 h2 ... ; h2 -h1 ...]
    hEq(1,:) = conj(hEq(1,:)); %  [h1* h2* ... ; h2 -h1 .... ]
    hEqPower = sum(hEq.*conj(hEq),1);

    yHat = sum(hEq.*yMod,1)./hEqPower; % [h1*y1 + h2y2*, h2*y1 -h1y2*, ... ]
    yHat(2:2:end) = conj(yHat(2:2:end));

    % receiver - hard decision decoding
    ipHat = real(yHat)>0;

    % counting the errors
    nErr(eee) = size(find((ip- ipHat)),2);    
    
    
    
end
toc

EbN0Lin = 10.^(SNRs/10);
p = 1/2 - 1/2*(1+1./EbN0Lin).^(-1/2);
theoryBer_nRx2 = p.^2.*(1+2*(1-p)); 

pAlamouti = 1/2 - 1/2*(1+2./EbN0Lin).^(-1/2);
theoryBerAlamouti_nTx2_nRx1 = pAlamouti.^2.*(1+2*(1-pAlamouti)); 

figure(1);
semilogy(SNRs, BERs_MRRC/(N*1.0),"bs-");
hold on;
semilogy(SNRs, BERs_Alamouti/(N*1.0),"rd-");
hold on;
%semilogy(SNRs,theoryBer_nRx2, "g*--");
semilogy(SNRs,theoryBerAlamouti_nTx2_nRx1, "g*-");
hold on;
semilogy(SNRs,theoryBer_nRx2, "ko-");
hold on;
semilogy(SNRs,nErr/(N*1.0), "y*-");

legend("MRRC","Alamouti","theory 21 Alamouti","theory MRRC","from dsplog");
grid on;