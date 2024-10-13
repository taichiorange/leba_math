clear;

tic;

Nt = 100;
Nr = 100;

N = 1e5;



pPlus = zeros(Nr,Nt) + 0.5;

sumLambda = zeros(Nt,1);
sumU = zeros(Nr,1);


xHat = zeros(Nt,1);

IterN = 5;
alpha = 0;

SNRs = (-10:1:10);

BERs = zeros(size(SNRs));

parfor ee = (1:length(SNRs)) 
    Lambda = zeros(Nr,Nt);
    sumSigma = zeros(Nr,1);

    SNR = SNRs(ee);
    Pnoise = 1/10^(SNR/10);
    Nerror = 0;
    
    for nn=(1:N)
        H = 1/sqrt(Nt) * (randn(Nr,Nt) + 1j * randn(Nr,Nt))/sqrt(2);
        Dnoise = sqrt(Pnoise/2) * (randn(Nr,1) + 1j * randn(Nr,1));
        x = 2*randi([0,1],[Nt,1]) -1;
        
        y = H*x + Dnoise;
        
        pPlus = zeros(Nr,Nt) + 0.5;
        
        Lambda(:,:) = 0;
        for t=(1:IterN)
            %观测节点，也就是接收端作为节点，计算 LLRs
            %%% 先把总体求和算出来，然后用逐个减去，可降低运算量
            HpPlus = H .* (2*pPlus -1);
            HHpPlus_pMinus = 4 * conj(H).*H .* pPlus.*(1-pPlus);
            sumU = sum(HpPlus,2);
            sumSigma = sum(HHpPlus_pMinus ,2);
            
            muZ = sumU - HpPlus;
            sigmaZ = sumSigma - HHpPlus_pMinus + Pnoise;
            Lambda = alpha * Lambda + (1-alpha)*4./sigmaZ .* real(conj(H) .*(y-muZ));
             
%             for i=(1:Nr)
%                 % 为每个接收节点，计算综合
%                   sumU(i) = 0;  % 第 i 个观测节点处算出来的均值相关的总和
% %                 sumSigma(i) = 0;
% %                 for j=(1:Nt)
% %                     sumU(i) = sumU(i) + H(i,j) * (2*pPlus(i,j) -1);
% %                     sumSigma(i) = sumSigma(i) + 4 * H(i,j)'*H(i,j) * pPlus(i,j)*(1-pPlus(i,j));
% %                 end
%                 %计算 Lambda, LLRs
%                 for k=(1:Nt)
%                     muZ = sumU(i) - H(i,k) * ( 2*pPlus(i,k)-1);
%                     sigmaZ = sumSigma(i) - 4 * H(i,k)'*H(i,k)*pPlus(i,k)*(1-pPlus(i,k)) + Pnoise;
%                     Lambda(i,k) = alpha * Lambda(i,k) + (1-alpha)*4/sigmaZ(i,k) * real(H(i,k)'*(y(i)-muZ(i,k)));
%                 end
%             end
            
            
            
            %计算变量节点的概率
            sumLambda = sum(Lambda,1);
            pL = exp(sumLambda - Lambda);
            pPlus = pL ./ ( 1+pL);
            
%             for k=(1:Nt)
%                 % 计算Lambda 的总和，方便后面计算，后面用减法
%                  sumLambda(k) = 0;
%                  for l=(1:Nr)
%                      sumLambda(k) = sumLambda(k) + Lambda(l,k);
%                  end
%                 
%                 %计算变量节点的概率信息，等于1的概率（不是 -1 的概率）
%                 for i=(1:Nr)
%                     pL = exp(sumLambda(k) - Lambda(i,k));  
%                     pPlus(i,k) = pL/(1+pL);
%                 end
%             end

        end
        
        %判决比特，计算错误数量并累加错误数量
        tL = sum(Lambda,1);
        xHat = 2*(tL>0) - 1;
        
        %%%%%累积错误数量
        Nerror = Nerror + sum(not(x==xHat'));
    end
    BERs(ee) = Nerror;
end
BERs = BERs/(N*Nt*1.0);
toc;

figure();
semilogy(SNRs,BERs,"b*--");
grid on;