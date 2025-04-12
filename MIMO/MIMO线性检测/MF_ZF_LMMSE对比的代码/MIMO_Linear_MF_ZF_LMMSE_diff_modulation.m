% 这是一个速度优化的版本，另外，调制使用 qam 调制


Nsymb = 1e5;
Ntx = 2;
Nrx = 2;
Rate = 4;
CN_db = 0;
vec_SNR = [-10:30];
vec_BER = [];

for ii=vec_SNR
    SNR = ii;
    
    H = 1/sqrt(Nt) *complex(randn(Ntx,Nrx,Nsymb),randn(Ntx,Nrx,Nsymb));
    
    NoiseVar = db2pow(-SNR);
    B = randi([0 2^Rate-1],Ntx,Nsymb);
    %X = 2*randi([0,1],Ntx,Nsymb) - 1;
    X = qammod(B,2^Rate,'UnitAveragePower',true);
    Z = complex(randn(Nrx,Nsymb),randn(Nrx,Nsymb))*sqrt(NoiseVar/2);

    X_reshaped = reshape(X, Ntx, 1, Nsymb); % → Ntx × 1 × Nsymb
    Y = squeeze(pagemtimes(H, X_reshaped)) + Z;
    SIGMA = NoiseVar * repmat(eye(Nrx), 1, 1, Nsymb);
    
    H_EST = H;
    G_ZF = pageinv(H);
    Hct = permute(conj(H), [2 1 3]);
    G_MMSE = pagemtimes(pageinv(pagemtimes(Hct,H)+SIGMA),Hct);
    
    Y = reshape(Y, Ntx, 1, Nsymb); % → Ntx × 1 × Nsymb
    X_ZF = squeeze(pagemtimes(G_ZF,Y));
    X_MMSE = squeeze(pagemtimes(G_MMSE,Y));
    
    % B_ZF = 2*(X_ZF>0)-1;
    % B_MMSE = 2*(X_MMSE>0)-1;
    % BER_ZF = sum(B_ZF~= X,"all")/Nsymb;
    % BER_MMSE = sum(B_MMSE~= X,"all")/Nsymb;

    
    B_ZF = qamdemod(X_ZF,2^Rate,'UnitAveragePower',true);
    B_MMSE = qamdemod(X_MMSE,2^Rate,'UnitAveragePower',true);
    
    Tmp=dec2bin(bitxor(B,B_ZF),Rate);
    BER_ZF = sum(sum(Tmp=='1'))/(length(B(:))*Rate);
    Tmp=dec2bin(bitxor(B,B_MMSE),Rate);
    BER_MMSE = sum(sum(Tmp=='1'))/(length(B(:))*Rate);

    vec_BER = [vec_BER;BER_ZF,BER_MMSE];
end
semilogy(vec_SNR,vec_BER(:,1),"--", vec_SNR, vec_BER(:,2),"-*");
legend("ZF", "MMSE");
grid on;