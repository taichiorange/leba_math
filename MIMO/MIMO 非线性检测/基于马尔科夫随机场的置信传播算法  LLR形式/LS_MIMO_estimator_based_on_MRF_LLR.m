clear;

Nt = 10;
Nr = 10;

N = 1e5;
IterN = 5;
alpha = 0.2;  % damping coefficients

SNRs=(-10:1:10);

BERs = zeros(size(SNRs));

tic

mij_LLR = zeros(Nt,Nt);
mij_LLR_new = zeros(Nt,Nt);
psi_xixj_LLR = zeros(4,Nt,Nt);
bi = zeros(Nt,1);

SQRT_NT = sqrt(Nt);

for eee=(1:length(SNRs))
    SNR = SNRs(eee);
    Pnoise = 1/10^(SNR/10);
    Nerror = 0;
    
    for nnn=(1:N)
        H = 1/SQRT_NT*(randn(Nr,Nt) + 1j * randn(Nr,Nt))/sqrt(2);
        
        Dnoise = sqrt(Pnoise/2) * (randn(Nr,1) + 1j * randn(Nr,1));
        x= 2*randi([0,1],[Nt,1]) - 1;
        y = H * x + Dnoise;
        
        % Initialization
        mij_LLR(:,:,:) = 0;
        
        R = 1/Pnoise * (H' * H);
        Z = 1/Pnoise * (H' * y);
        

        
%         for i=(1:Nt)
%             phi_xi(1,i) = exp((-1) * real(Z(i)) + log(0.5));   % for -1
%             phi_xi(2,i) = exp((+1) * real(Z(i)) + log(0.5));   % for +1
%         end
        phi_x_LLR =  real(Z) - (-1)*real(Z);   % for (xi=1)-(xi=-1)
        
%         for i=(1:Nt)
%             for j=(1:Nt)
%                 psi_xixj(1,i,j) = exp(-(-1)*real(R(i,j))*(-1));  % -1  -1
%                 psi_xixj(2,i,j) = exp(-(-1)*real(R(i,j))*(+1));  % -1  +1
%                 psi_xixj(3,i,j) = exp(-(+1)*real(R(i,j))*(-1));  % +1  -1
%                 psi_xixj(4,i,j) = exp(-(+1)*real(R(i,j))*(+1));  % +1  +1
%             end
%         end
        psi_xixj_LLR(1,:,:) = -(-1)*real(R)*(-1);  % -1  -1
        psi_xixj_LLR(2,:,:) = -(-1)*real(R)*(+1);  % -1  +1
        psi_xixj_LLR(3,:,:) = -(+1)*real(R)*(-1);  % +1  -1
        psi_xixj_LLR(4,:,:) = -(+1)*real(R)*(+1);  % +1  +1        

        % Iterative update of messages
        for t=(1:IterN)
            
            %message calculation : i--->j
            for j=(1:Nt)
                for i=(1:Nt)
                    if j==i
                        continue;
                    end
                    
                    % sum part
                    mki_xi = 0;  % mki_llr sum
                    for k=(1:Nt)
                        if k==j
                            continue;
                        end
                        mki_xi = mki_xi + mij_LLR(k,i);
                    end
                    
                    % sum part
                    %%% xj = -1
                    max_m1 = max(psi_xixj_LLR(3,i,j)+phi_x_LLR(i), psi_xixj_LLR(1,i,j));
                    mij_pie_m1 = exp(phi_x_LLR(i) + psi_xixj_LLR(3,i,j)- max_m1 + mki_xi) + ... % xi = -1 , xj = -1
                                 exp(psi_xixj_LLR(1,i,j) - max_m1);        % xi = +1 , xj = -1
                    %%% xj = 1
                    max_1 = max(psi_xixj_LLR(4,i,j)+phi_x_LLR(i), psi_xixj_LLR(2,i,j) );
                    mij_pie_1 = exp(phi_x_LLR(i) + psi_xixj_LLR(4,i,j) - max_1 + mki_xi) + ... % xi = -1 , xj = +1
                                exp(psi_xixj_LLR(2,i,j) - max_1);        % xi = +1 , xj = +1
                    LLR_m = max_1 - max_m1 + log(mij_pie_1/mij_pie_m1);
                    
                    mij_LLR_new(i,j) = LLR_m;

                end
            end
            % damping messages
            mij_new_damp = 1./(1+exp(-mij_LLR_new));
            mij_old_damp = 1./(1+exp(-mij_LLR));
            
            mij_LLR(:,:) = log(alpha * mij_old_damp + (1-alpha) * mij_new_damp) -  ...
                            log(alpha * (1-mij_old_damp) + (1-alpha) * (1-mij_new_damp));
        end
        
        % Belief calculation
%         for i=(1:Nt)
%             % product part
% %             bi_t = 0;
% %             for j=(1:Nt)
% %                 bi_t = bi_t + mij_LLR(j,i);  
% %             end
%             bi_t = sum(mij_LLR(:,i));
%             % belief
%             bi(i) = phi_x_LLR(i) + bi_t;    % xi = -1;
%         end
        
        bi = sum(mij_LLR,1)' + phi_x_LLR;
        
        xi = 2*(bi>0)-1;
        
        Nerror = Nerror + sum(not(x==xi));
    end
    BERs(eee) = Nerror;
end
toc

BERs_ratio = BERs / (N*Nt*1.0);

figure();
semilogy(SNRs, BERs_ratio);
% ylim([0.001,1]);
grid on;

                