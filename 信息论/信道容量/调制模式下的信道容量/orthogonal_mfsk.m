%Performance of baseband M-orthogonal modulation scheme
clearvars; clc;
%---------Input Fields------------------------
N=100000;%Number of symbols to transmit
EbN0dB = -4:2:14; % Eb/N0 range in dB for simulation
arrayofM=[8]; %M-level number of orthogonal basis functions
detectionType = 'NON_COHERENT';%valid : COHERENT/NON_COHERENT
%----------------------------------------------
plotColor =['b','g','r','c','m','k']; %For plotting
p=1; %For plot color index
legendStr={'Sim M=2','Theory M=2','Sim M=4','Theory M=4','Sim M=8','Theory M=8','Sim M=16','Theory M=16','Sim M=32','Theory M=32',};

for M = arrayofM %run this loop for each value of M
  k=log2(M); %Bits per symbol    
  %-----------Data generation---------------------
  abits=rand(1,N*k)>=0.5; %random information bits
  abitArray=reshape(abits,k,N).'; %each row holds one symbol
  factors=2.^(numel(abitArray(1,:))-1:-1:0);%for binary to decimal
  aSyms=(abitArray*factors.').';%binary to integer - Symbols
  d=ceil(M.*rand(1,N));

  %--------------Modulation-------------------
  [s,ref]= mfsk_modulator(FSK_TYPE,M,d);

  EbN0lin = 10.^(EbN0dB/10);%Converting to linear scale
  EsN0lin=k*EbN0lin; %Converting Eb/N0 to Es/N0
  SER = zeros(length(EsN0lin),1); %Place holder for SER values
  BER = SER; Ps=SER; Pb=SER; %Place holders
  q=1;

  for EsN0=EsN0lin,
      %Adding noise with variance according to the required Es/N0
      Es=sum(sum(abs(s).^2))/(length(s)); %actual symbol energy
      N0=Es/EsN0; %Find the noise spectral density
      noiseSigma = sqrt(N0/2); %standard deviation for AWGN Noise
        
      if strcmpi(detectionType,'COHERENT'),
         noise = noiseSigma*(randn(N,M));%real values since phi=0
      else
         noise = noiseSigma*(randn(N,M)+1i*randn(N,M));%complex noise
      end

      y=s+noise;  %Channel - Adding AWGN Noise

      [dCap]= mfsk_detector(COHERENCE,r,ref);
        
      SER(q)=sum((aSyms~=aCapSyms))/N; %Compute Symbol Error Rate
      aCapBits=dec2bin(aCapSyms-1) - '0';%-1 added for Matlab index
      aCapBits=reshape(aCapBits.',1,N*k);%sequence of decoded bits
      BER(q)=sum((abits~=aCapBits))/(N*k); %Compute Bit Error Rate

      if strcmpi(detectionType,'NON_COHERENT'),
         %compute theoretical BER only for non coherent case
         summ=0;
         for i=1:M-1,
           n=M-1; r=i; %for nCr formula
           summ=summ+(-1).^(i+1)./(i+1).*prod((n-r+1:n)./(1:r)).*exp(-i./(i+1).*EsN0);
         end
         Ps(q)=summ; %Theoretical BER for non-coherent detection
         Pb(q)=M/(2*M-2)*Ps(q);%Theoretical Bit error probability
      end        
      q=q+1;
      semilogy(EbN0dB,BER,[plotColor(p) '*']); hold on;    
        
      if strcmpi(detectionType,'NON_COHERENT'), 
        semilogy(EbN0dB,Pb,[plotColor(p) '-']); 
      else
        disp('Use BERTOOL/numerical methods for theoretical BER');
      end
  end
      p=p+1;    
end
legend(legendStr);xlabel('Eb/N0(dB)');ylabel('Bit Error Rate (Pb)');
title('Probability of Bit Error for M-Orthogonal signals');grid on;