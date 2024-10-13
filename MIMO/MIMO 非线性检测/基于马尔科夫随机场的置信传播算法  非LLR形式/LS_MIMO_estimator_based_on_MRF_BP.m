clear;

Nt = 10;
Nr = 10;

N = 1e5;
IterN = 5;
alpha = 0.2;  % damping coefficients

SNRs=(-10:1:10);

BERs = zeros(size(SNRs));

tic

mij = zeros(2,Nt,Nt);
mij_new = zeros(2,Nt,Nt);
phi_xi = zeros(2,Nt);
psi_xixj = zeros(4,Nt,Nt);
bi = zeros(2,Nt);
xi = zeros(1,Nt); 

for eee=(1:length(SNRs))
    SNR = SNRs(eee);
    Pnoise = 1/10^(SNR/10);
    Nerror = 0;
    
    for nnn=(1:N)
        H = 1/sqrt(Nt)*(randn(Nr,Nt) + 1j * randn(Nr,Nt))/sqrt(2);
        
        Dnoise = sqrt(Pnoise/2) * (randn(Nr,1) + 1j * randn(Nr,1));
        x= 2*randi([0,1],[Nt,1]) - 1;
        y = H * x + Dnoise;
        
        % Initialization
        mij(:,:,:) = 0.5;
        mij_new(:,:,:) = 0.5;

        
        R = 1/Pnoise * (H' * H);
        Z = 1/Pnoise * (H' * y);
        

        
%         for i=(1:Nt)
%             phi_xi(1,i) = exp((-1) * real(Z(i)) + log(0.5));   % for -1
%             phi_xi(2,i) = exp((+1) * real(Z(i)) + log(0.5));   % for +1
%         end
        phi_xi(1,:) =  exp((-1) * real(Z) + log(0.5));   % for -1
        phi_xi(2,:) = exp((+1) * real(Z) + log(0.5));   % for +1
        
%         for i=(1:Nt)
%             for j=(1:Nt)
%                 psi_xixj(1,i,j) = exp(-(-1)*real(R(i,j))*(-1));  % -1  -1
%                 psi_xixj(2,i,j) = exp(-(-1)*real(R(i,j))*(+1));  % -1  +1
%                 psi_xixj(3,i,j) = exp(-(+1)*real(R(i,j))*(-1));  % +1  -1
%                 psi_xixj(4,i,j) = exp(-(+1)*real(R(i,j))*(+1));  % +1  +1
%             end
%         end
        psi_xixj(1,:,:) = exp(-(-1)*real(R)*(-1));  % -1  -1
        psi_xixj(2,:,:) = exp(-(-1)*real(R)*(+1));  % -1  +1
        psi_xixj(3,:,:) = exp(-(+1)*real(R)*(-1));  % +1  -1
        psi_xixj(4,:,:) = exp(-(+1)*real(R)*(+1));  % +1  +1        

        % Iterative update of messages
        for t=(1:IterN)
            
            %message calculation : i--->j
            for i=(1:Nt)
                for j=(1:Nt)
                    if j==i
                        continue;
                    end
                    
                    % product part
                    mki_xi_m1 = 1;  % mki product ofr xi = -1
                    mki_xi1 = 1;     % mki product for xi=1
                    for k=(1:Nt)
                        if k==j
                            continue;
                        end
                        mki_xi_m1 = mki_xi_m1 * mij(1,k,i); % xi = -1
                        mki_xi1 = mki_xi1 * mij(2,k,i); % xi = 1;
                    end
                    
                    % sum part
                    %%% xj = -1
                    mij_pie_m1 = phi_xi(1,i) * psi_xixj(1,i,j)*mki_xi_m1 + ... % xi = -1 , xj = -1
                                 phi_xi(2,i) * psi_xixj(3,i,j)*mki_xi1;        % xi = +1 , xj = -1
                    %%% xj = 1
                    mij_pie_1 = phi_xi(1,i) * psi_xixj(2,i,j)*mki_xi_m1 + ... % xi = -1 , xj = +1
                                 phi_xi(2,i) * psi_xixj(4,i,j)*mki_xi1;        % xi = +1 , xj = +1
                             
                    % message normalization
                    mij_pie_m1 = mij_pie_m1/(mij_pie_m1 + mij_pie_1);
                    mij_pie_1 = 1 - mij_pie_m1;
                    
                    % damping messages
                    mij_pie_m1 = alpha * mij(1,i,j) + (1-alpha) * mij_pie_m1;   % xj = -1
                    mij_pie_1 =  alpha * mij(2,i,j) + (1-alpha) * mij_pie_1;    % xj = 1
                    

                    mij_new(1,i,j) = mij_pie_m1;
                    mij_new(2,i,j) = mij_pie_1;
                end
            end
            
            mij(:,:,:) = mij_new(:,:,:);
        end
        
        % Belief calculation
        for i=(1:Nt)
            % product part
            bi_m1 = 1;
            bi_1 = 1;
            for j=(1:Nt)
                bi_m1 = bi_m1 * mij(1,j,i);  % xi = -1
                bi_1 = bi_1 * mij(2,j,i);   % xi = 1
            end
            % belief
            bi(1,i) = phi_xi(1,i) * bi_m1;    % xi = -1;
            bi(2,i) = phi_xi(2,i) * bi_1;     % xi = 1
%             if bi(1,i) > bi(2,i)
%                 xi(i) = -1;
%             else
%                 xi(i) = 1;
%             end
        end
        
        xi = 2*(bi(1,:)<bi(2,:))-1;
        
        Nerror = Nerror + sum(not(x'==xi));
    end
    BERs(eee) = Nerror;
end
toc

BERs_ratio = BERs / (N*Nt*1.0);

figure();
semilogy(SNRs, BERs_ratio);
ylim([0.001,1]);
grid on;

                