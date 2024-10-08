clear;

SNRs = -10:2:30;

N = 1e6;

Vars = [3,2,1,1/2,1/3,1/4];


BERs = zeros(length(Vars), length(SNRs));
BERs_theory= zeros(length(Vars), length(SNRs));

CC = [0,1]';

XX = 2*CC - 1;

for j=1:length(Vars)
    var = Vars(j);

    % theory BER
    SNRs_L = 10.^(SNRs/10);
    sqrtSNR = (SNRs_L./(1/var+SNRs_L)).^(1/2);
    BERs_theory(j,:) = 1/2 * (1-sqrtSNR);
    
    tic
    for i=1:length(SNRs)
        
        snr = SNRs(i);
        Pnoise = 1/10^(snr/10);
    
        NerrorML = 0;
        for ii=(1:N)
            c = randi([0,1],1);
            x = 2*c - 1;
            H = (randn(1) + 1j*randn(1))*sqrt(var);
            n = sqrt(Pnoise) * (randn(1) + 1j * randn(1));
            
            y = H*x + n;

       
            % Maximum Likelihood
            [val,index]=min(abs(y - H*XX).^2);
            c_hat = CC(index,1);
        
            NerrorML = NerrorML + sum(c_hat ~= c,'all');
    %             yHat = y./H;
    %             ipHat = real(yHat)>0;
    %             NerrorML = NerrorML + sum(ipHat ~= c,'all');
    
        end
    
        BERs(j,i) = NerrorML/(N);
    
    end
    toc
end


% AWGN channel, no fading

BERs_AWGN = zeros(size(SNRs));


% theory BER
tic
for i=1:length(SNRs)
    
    snr = SNRs(i);
    Pnoise = 1/10^(snr/10);

    NerrorML = 0;
    for ii=(1:N)
        c = randi([0,1],1);
        x = 2*c - 1;
        H = 1;
        n = sqrt(Pnoise) * (randn(1) + 1j * randn(1));
        
        y = H*x + n;
        
   
        % Maximum Likelihood
        [val,index]=min(abs(y - H*XX).^2);
        c_hat = CC(index,1);
    
        NerrorML = NerrorML + sum(c_hat ~= c,'all');
%             yHat = y./H;
%             ipHat = real(yHat)>0;
%             NerrorML = NerrorML + sum(ipHat ~= c,'all');

    end

    BERs_AWGN (i) = NerrorML/(N);

end
toc

figure;
semilogy(SNRs,BERs(1,:),"*--");
hold on;
semilogy(SNRs,BERs(2,:),"--");
hold on;
semilogy(SNRs,BERs(3,:),"--");
hold on;
semilogy(SNRs,BERs(4,:),"--");
hold on;
semilogy(SNRs,BERs(5,:),"--");
hold on;
semilogy(SNRs,BERs(6,:),"d--");
hold on;

semilogy(SNRs,BERs_theory(1,:),"*-");
hold on;
semilogy(SNRs,BERs_theory(2,:));
hold on;
semilogy(SNRs,BERs_theory(3,:));
hold on;
semilogy(SNRs,BERs_theory(4,:));
hold on;
semilogy(SNRs,BERs_theory(5,:));
hold on;
semilogy(SNRs,BERs_theory(6,:),"d-");
hold on;

semilogy(SNRs,BERs_AWGN,"r*--");
hold on;

legend("sim var=3","sim var=2","sim var=1","sim var=1/2","sim var=1/3","sim var=1/4", ...
    "thry var=3","thry var=2","thry var=1","thry var=1/2","thry var=1/3","thry var=1/4",  "AWGN only");

grid on;


% save('MIMO_detection_MaximumLikelihood_4x4.mat','SNRs');
% save('MIMO_detection_MaximumLikelihood_4x4.mat','ErrCnts','-append');
% save('MIMO_detection_MaximumLikelihood_4x4.mat','BERs','-append');

