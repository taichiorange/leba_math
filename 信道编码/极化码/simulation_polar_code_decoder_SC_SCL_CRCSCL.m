clear;
crc_len = 4;  %CRCУ��λ����
n = 8; % n = 8;
N = 2^n; %�����볤��
R=0.5;   %����
max_runs = 1000; %������д���
msgbit_len = N*R;   %������Ϣ���صĳ���
K = msgbit_len+crc_len; %����λ����
snr=2;   %��˹�ŵ������
L=16;  %CASCL���������б���
[gen, det] = get_crc_objective(crc_len);%�����Ӧ��CRC��������CRCУ����

%�Բ������б����Ա����ظ�����
lambda_offset = 2.^(0 : log2(N)); %�ֶ�����
llr_layer_vec = get_llr_layer(N); %LLR����ʵ��ִ�в�������
bit_layer_vec = get_bit_layer(N); %����ֵ����ʱʵ��ִ�в�������

%info = rand(msgbit_len, 1) > 0.5;%���������Ϣ���أ��������0.5��Ƚ϶��õ��߼�ֵ
info=randsrc(msgbit_len,1,[0 1;0.5 0.5]); %�ȸ���������Ϣ���� 
%ͨ���Աȿ�֪���ַ�ʽ��������ɻ�ø��õ�����Ч��
% info_with_crc = [info; mod(crc_parity_check * info, 2)];
info_with_crc=generate(gen,info);

%Gaussian approximation Code Constructionsnr = 2.5;
sigma = 1/sqrt(2 * R) * 10^(-snr/20); %��˹�ŵ�����ȷ���
channels= GA(sigma, N); %��˹���ƹ���

[~, channel_ordered] = sort(channels, 'descend');  %���ŵ��������ϰ��ս�������
info_bits = sort(channel_ordered(1 : K), 'ascend'); %����Ϣλ�������ϰ���������
frozen_bits = ones(N , 1); 
frozen_bits(info_bits) = 0; %ʹ��ϢλΪ0���������Ϊ1��
logic=mod(frozen_bits + 1, 2); %����������ֵȡ��
info_bits_logical = logical(logic); %����ֵת��Ϊ�߼�ֵ
u = zeros(N, 1);
u(info_bits_logical) = info_with_crc; %����Ϣ����ӳ������Դ��������Ϣλ����
x = polar_encoder(u); 
bpsk = 1 - 2 * x;  
%ѭ����ʼ
blenum_sc=0;%��������ʼ��
blenum_scl=0;
blenum_cascl=0;
for i_run = 1 : max_runs
        y=awgn(bpsk,snr);
        llr = 2/sigma^2*y;
        polar_info_sc = SC_decoder(llr, K, frozen_bits, lambda_offset, llr_layer_vec, bit_layer_vec);
        polar_info_scl = SCL_decoder(llr, L, K, frozen_bits, lambda_offset, llr_layer_vec, bit_layer_vec);
        polar_info_cascl = CASCL_decoder(llr, L, K, frozen_bits, det, lambda_offset, llr_layer_vec, bit_layer_vec);
       if any(polar_info_sc ~= info_with_crc)
           blenum_sc= blenum_sc + 1; %����ʧ���ۼ�һ��
       end
       if any(polar_info_scl ~= info_with_crc)
           blenum_scl= blenum_scl + 1;
       end
       if any(polar_info_cascl ~= info_with_crc)
           blenum_cascl= blenum_cascl + 1;
       end
      
end
bler_sc=blenum_sc/max_runs; %���������
bler_scl=blenum_scl/max_runs;
bler_cascl=blenum_cascl/max_runs;
fprintf('Sim iteration running = %d\n',max_runs);%����������д���
fprintf('N = %d,K = %d,L = %d\n',N,K,L);
fprintf('the SNR = %.1f\n',snr);
fprintf('the BLER of SC = %f\n',bler_sc);
fprintf('the BLERs of SCL = %f\n',bler_scl);
fprintf('the BLERs of CA-SCL = %f\n',bler_cascl);



function polar_info_esti = SC_decoder(llr,K, frozen_bits, lambda_offset, llr_layer_vec, bit_layer_vec)
N = length(llr);%llr refers to channel LLR.
n = log2(N);
P = zeros(N - 1, 1);%channel llr is not include in P.
C = zeros(N - 1, 2)-1;%C stores internal bit values
polar_info_esti = zeros(K, 1);
cnt_K = 1;
for phi = 0 : N - 1
    switch phi
        case 0%for decoding u_1      ������һ��ĵ�һ��
            index_1 = lambda_offset(n);  % index_1 �� 4�� ��� N=8 �Ļ�
            % ��� for ѭ�������㵹���ڶ���� LLR ��Ȼ�ȣ�N=8�����룬������� 4 �� LLR
            % P(4) <==  LLR(1) �� LLR(5)
            % P(5) <==  LLR(2) �� LLR(6)
            % P(6) <==  LLR(3) �� LLR(7)
            % P(7) <==  LLR(4) �� LLR(8)
            for beta = 0 : index_1 - 1%use llr vector
                P(beta + index_1) =  sign(llr(beta + 1)) * sign(llr(beta + 1 + index_1)) * min(abs(llr(beta + 1)), abs(llr(beta + 1 + index_1)));
            end
            % ��� for ѭ����������˵����ڶ���������� LLR ��Ȼ�ȣ�ѭ�����Σ��ֱ���㶼�������(2�����͵��Ĳ㣨1������ LLR
            % i_layer = 1 ʱ, index_1=2  index_2 = 4
            %    P(2)  <== P(4) �� P(6)
            %    P(3)  <== P(5) �� P(7)
            % i_layer = 0 ʱ, index_1=1  index_2 = 2
            %    P(1)  <== P(2) �� P(3)
            for i_layer = n - 2 : -1 : 0%use P vector
                index_1 = lambda_offset(i_layer + 1);
                index_2 = lambda_offset(i_layer + 2);
                for beta = index_1 : index_2 - 1
                    P(beta) =  sign(P(beta + index_1)) * sign(P(beta + index_2)) * min(abs(P(beta + index_1)), abs(P(beta + index_2)));
                end
            end
        case N/2%for deocding u_{N/2 + 1}   ������һ��ĵ�һ��
            index_1 = lambda_offset(n);
            % ��һ��ĵ����ڶ��㣬����϶�����������ı���
            for beta = 0 : index_1 - 1%use llr vector. g function.
                P(beta + index_1) = (1 - 2 * C(beta + index_1, 1)) * llr(beta + 1) + llr(beta + 1 + index_1);
            end
            % ���������ϣ����󣩸������Ȼ�ȣ����û����������ı��ص����
            for i_layer = n - 2 : -1 : 0%use P vector. f function
                index_1 = lambda_offset(i_layer + 1);
                index_2 = lambda_offset(i_layer + 2);
                for beta = index_1 : index_2 - 1
                    P(beta) =  sign(P(beta + index_1)) * sign(P(beta + index_2)) * min(abs(P(beta + index_1)), abs(P(beta + index_2)));
                end
            end
        otherwise
            llr_layer = llr_layer_vec(phi + 1);
            index_1 = lambda_offset(llr_layer + 1);
            index_2 = lambda_offset(llr_layer + 2);
            % ����Ǽ���ǰ���Ѿ���������� u ��N=2������� ��Ȼ��
            for beta = index_1 : index_2 - 1%g function is first implemented.
                P(beta) = (1 - 2 * C(beta, 1)) * P(beta + index_1) + P(beta + index_2);
            end
            % ���� N=2 �������û����������ص�����������ж��
            for i_layer = llr_layer - 1 : -1 : 0%then f function is implemented.
                index_1 = lambda_offset(i_layer + 1);
                index_2 = lambda_offset(i_layer + 2);
                for beta = index_1 : index_2 - 1
                    P(beta) =  sign(P(beta + index_1)) * sign(P(beta + index_2)) * min(abs(P(beta + index_1)), abs(P(beta + index_2)));
                end
            end
    end
    phi_mod_2 = mod(phi, 2);
    if frozen_bits(phi + 1) == 1%frozen bit
        C(1, 1 + phi_mod_2) = 0;
    else%information bit
        C(1, 1 + phi_mod_2) = P(1) < 0;%store internal bit values
        polar_info_esti(cnt_K) = P(1) < 0;
        cnt_K = cnt_K + 1;
    end
    if phi_mod_2  == 1 && phi ~= N - 1
        bit_layer = bit_layer_vec(phi + 1);
        for i_layer = 0 : bit_layer - 1%give values to the 2nd column of C
            index_1 = lambda_offset(i_layer + 1);
            index_2 = lambda_offset(i_layer + 2);
            for beta = index_1 : index_2 - 1
                C(beta + index_1, 2) = mod(C(beta, 1) + C(beta, 2), 2);
                C(beta + index_2, 2) = C(beta, 2);
            end
        end
        index_1 = lambda_offset(bit_layer + 1);
        index_2 = lambda_offset(bit_layer + 2);
        for beta = index_1 : index_2 - 1%give values to the 1st column of C
            C(beta + index_1, 1) = mod(C(beta, 1) + C(beta, 2), 2);
            C(beta + index_2, 1) = C(beta, 2);
        end
    end
end
end



function layer_vec = get_llr_layer(N)
layer_vec = zeros(N , 1);
for phi = 1 : N - 1
    psi = phi;
    layer = 0;
    while(mod(psi, 2) == 0)
        psi = floor(psi/2);
        layer = layer + 1;
    end
    layer_vec(phi + 1) = layer;
end
end


function layer_vec = get_bit_layer(N)
layer_vec = zeros(N, 1);
for phi = 0 : N - 1
    psi = floor(phi/2);
    layer = 0;
    while(mod(psi, 2) == 1)
        psi = floor(psi/2);
        layer = layer + 1;
    end
    layer_vec(phi + 1) = layer;
end
end


function u = GA(sigma, N)
% N=512;
% sigma=0.5;
u = zeros(1, N);
u(1) = 2/sigma^2;
for i = 1:log2(N)
    j = 2^(i - 1);
    for k = 1:j
        tmp = u(k);
        u(k) = phi_inverse(1 - (1 - phi(tmp))^2);
        u(k + j) = 2 * tmp;
    end
end
u = bitrevorder(u);

% scatter((1:N),u(1:N),'.b');
% axis([0 1.1*N 0 4*N]);
% xlabel('Channel index');
% ylabel('E(LLRi)');
end



function [gen, det] = get_crc_objective(crc_length)
    switch crc_length
        case 4
            gen = crc.generator('Polynomial',[1 0 0 1 1],'InitialState',zeros(1, 4),'FinalXOR',zeros(1, 4));
            det = crc.detector('Polynomial',[1 0 0 1 1],'InitialState',zeros(1, 4),'FinalXOR',zeros(1, 4));
        case 6
            gen = crc.generator('Polynomial',[1 0 0 0 0 1 1],'InitialState',zeros(1, 6),'FinalXOR',zeros(1, 6));
            det = crc.detector('Polynomial',[1 0 0 0 0 1 1],'InitialState',zeros(1, 6),'FinalXOR',zeros(1, 6));
        case 8
            gen = crc.generator('Polynomial','0xA6','InitialState','0x00','FinalXOR','0x00');
            det = crc.detector('Polynomial','0xA6','InitialState','0x00','FinalXOR','0x00');
        case 10
            gen = crc.generator('Polynomial',[1 1 0 0 1 0 0 1 1 1 1],'InitialState',zeros(1, 10),'FinalXOR',zeros(1, 10));
            det = crc.detector('Polynomial',[1 1 0 0 1 0 0 1 1 1 1],'InitialState',zeros(1, 10),'FinalXOR',zeros(1, 10));
        case 12
            gen = crc.generator('Polynomial',[1 1 0 0 0 0 0 0 0 1 1 0 1],'InitialState',zeros(1, 12),'FinalXOR',zeros(1, 12));
            det = crc.detector('Polynomial',[1 1 0 0 0 0 0 0 0 1 1 0 1],'InitialState',zeros(1, 12),'FinalXOR',zeros(1, 12));
        case 16
            gen = crc.generator('Polynomial',[1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1],'InitialState',[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],'FinalXOR',[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]);
            det = crc.detector('Polynomial',[1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1],'InitialState',[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],'FinalXOR',[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]);
            g = [1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1];
        case 24
            gen = crc.generator('Polynomial',[1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1],'InitialState',[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],...
                'FinalXOR',[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]);
            det = crc.detector('Polynomial',[1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1],'InitialState',[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],...
                'FinalXOR',[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]);
        otherwise
            disp('Unsupported CRC length. Program terminates')
    end
end



function x = phi_inverse(y)
%�����ñպϱ��ʽ����������ֵ�ⷨ���ٶȽ�һ��������
if (y <= 1.0221) && (y >= 0.0388)
    x = ((0.0218 - log(y))/0.4527)^(1/0.86);
else
    x0 = 0.0388;
    x1 = x0 - (phi(x0) - y)/derivative_phi(x0);
    delta = abs(x1 - x0);
    epsilon = 1e-3;
    
    while(delta >= epsilon)
        x0 = x1;
        x1 = x1 - (phi(x1) - y)/derivative_phi(x1);
        %��x1���󣬷ſ�epsilon
        if x1 > 1e2
            epsilon = 10;
        end       
        delta = abs(x1 - x0);
    end
    x = x1;
end
end


function y = phi(x)
if (x >= 0)&&(x <= 10)
    y = exp(-0.4527*x^0.859 + 0.0218);
else
    y = sqrt(pi/x) * exp(-x/4) * (1 - 10/7/x);
end
end


function dx = derivative_phi(x)
if (x >= 0)&&(x <= 10)
    dx = -0.4527*0.86*x^(-0.14)*phi(x);
else
    dx = exp(-x/4)*sqrt(pi/x)*(-1/2/x*(1 - 10/7/x) - 1/4*(1 - 10/7/x) + 10/7/x/x);
end
end


function x = polar_encoder(u)
%encoding: x = u * Fn.
N = length(u);
GN=get_GN(N); 
Y=u'*GN;
x=mod(Y',2);
end


function FN = get_GN(N)
F = [1, 0 ; 1, 1];
FN = zeros(N, N);
FN(1 : 2, 1 : 2) = F;
for i = 2 : log2(N)
    FN(1 : 2^i, 1 : 2^i) = kron(FN(1 : 2^(i - 1), 1 : 2^(i - 1)), F);
end
end



function polar_info_esti = SCL_decoder(llr, L, K, frozen_bits, lambda_offset, llr_layer_vec, bit_layer_vec)
%LLR-based SCL deocoder, a single function, no other sub-functions.
%Frequently calling sub-functions will derease the efficiency of MATLAB
%codes.
%const
N = length(llr);
m = log2(N);
%memory declared
lazy_copy = zeros(m, L);%If Lazy Copy is used, there is no data-copy in the decoding process. We only need to record where the data come from. Here,data refer to LLRs and partial sums.
%Lazy Copy is a relatively sophisticated operation for new learners of polar codes. If you do not understand such operation, you can directly copy data.
%If you can understand lazy copy and you just start learning polar codes
%for just fews days, you are very clever,
P = zeros(N - 1, L); %Channel llr is public-used, so N - 1 is enough.
C = zeros(N - 1, 2 * L)-1;%I do not esitimate (x1, x2, ... , xN), so N - 1 is enough.
u = zeros(K, L);%unfrozen bits that polar codes carry, including crc bits.
PM = zeros(L, 1);%Path metrics
activepath = zeros(L, 1);%Indicate if the path is active. '1'��active; '0' otherwise.
cnt_u = 1;%information bit counter 
%initialize
activepath(1) = 1;
lazy_copy(:, 1) = 1;
%decoding starts
%default: in the case of path clone, the origianl path always corresponds to bit 0, while the new path bit 1.
for phi = 0 : N - 1
    layer = llr_layer_vec(phi + 1);
    phi_mod_2 = mod(phi, 2);
    for l_index = 1 : L
        if activepath(l_index) == 0
            continue;
        end
        switch phi%Decoding bits u_0 and u_N/2 needs channel LLR, so the decoding of them is separated from other bits. 
            case 0
                index_1 = lambda_offset(m);
                for beta = 0 : index_1 - 1
                    P(beta + index_1, l_index) = sign(llr(beta + 1)) * sign(llr(beta + index_1 + 1)) * min(abs(llr(beta + 1)), abs(llr(beta + index_1 + 1)));
                end
                for i_layer = m - 2 : -1 : 0
                    index_1 = lambda_offset(i_layer + 1);
                    index_2 = lambda_offset(i_layer + 2);
                    for beta = 0 : index_1 - 1
                        P(beta + index_1, l_index) = sign(P(beta + index_2, l_index)) *...
                            sign(P(beta + index_1 + index_2, l_index)) * min(abs(P(beta + index_2, l_index)), abs(P(beta + index_1 + index_2, l_index)));
                    end
                end
            case N/2
                index_1 = lambda_offset(m);
                for beta = 0 : index_1 - 1
                    x_tmp = C(beta + index_1, 2 * l_index - 1);
                    P(beta + index_1, l_index) = (1 - 2 * x_tmp) * llr(beta + 1) + llr(beta + 1 + index_1);
                end
                for i_layer = m - 2 : -1 : 0
                    index_1 = lambda_offset(i_layer + 1);
                    index_2 = lambda_offset(i_layer + 2);
                    for beta = 0 : index_1 - 1
                        P(beta + index_1, l_index) = sign(P(beta + index_2, l_index)) *...
                            sign(P(beta + index_1 + index_2, l_index)) * min(abs(P(beta + index_2, l_index)), abs(P(beta + index_1 + index_2, l_index)));
                    end
                end
            otherwise
                index_1 = lambda_offset(layer + 1);
                index_2 = lambda_offset(layer + 2);
                for beta = 0 : index_1 - 1
                    P(beta + index_1, l_index) = (1 - 2 * C(beta + index_1, 2 * l_index - 1)) * P(beta + index_2, lazy_copy(layer + 2, l_index)) +...
                        P(beta + index_1 + index_2, lazy_copy(layer + 2, l_index));
                end
                for i_layer = layer - 1 : -1 : 0
                    index_1 = lambda_offset(i_layer + 1);
                    index_2 = lambda_offset(i_layer + 2);
                    for beta = 0 : index_1 - 1
                        P(beta + index_1, l_index) = sign(P(beta + index_2, l_index)) *...
                            sign(P(beta + index_1 + index_2, l_index)) * min(abs(P(beta + index_2, l_index)),...
                            abs(P(beta + index_1 + index_2, l_index)));
                    end
                end
        end
    end
    if frozen_bits(phi + 1) == 0%if now we decode an unfrozen bit
        PM_pair = realmax * ones(2, L);
        for l_index = 1 : L
            if activepath(l_index) == 0
                continue;
            end
            if P(1, l_index) >= 0
                PM_pair(1, l_index) = PM(l_index);
                PM_pair(2, l_index) = PM(l_index) + P(1, l_index);
            else
                PM_pair(1, l_index) = PM(l_index) - P(1, l_index);
                PM_pair(2, l_index) = PM(l_index);
            end
        end
        middle = min(2 * sum(activepath), L);
        PM_sort = sort(PM_pair(:));
        PM_cv = PM_sort(middle);
        compare = PM_pair <= PM_cv; 
        kill_index = zeros(L, 1);%to record the index of the path that is killed
        kill_cnt = 0;%the total number of killed path
        %the above two variables consist of a stack
        for i = 1 : L
            if (compare(1, i) == 0)&&(compare(2, i) == 0)%which indicates that this path should be killed
                activepath(i) = 0;
                kill_cnt = kill_cnt + 1;%push stack
                kill_index(kill_cnt) = i;
            end
        end
        for l_index = 1 : L
            if activepath(l_index) == 0
                continue;
            end
            path_state = compare(1, l_index) * 2 + compare(2, l_index);
            %path_state: ��ǰ·�� l_index ���³���������·�������ֿ��ܣ���ǰ����ı���ȡ0����1��
            %   1����������Ϊ1��·��
            %   2����������Ϊ0��·��
            %   3����������·��
            %   0��������
            switch path_state%path_state can equal to 0, but in this case we do no operation.
                case 1
                    u(cnt_u, l_index) = 1;
                    C(1, 2 * l_index - 1 + phi_mod_2) = 1;
                    PM(l_index) = PM_pair(2, l_index);
                case 2
                    u(cnt_u, l_index) = 0;
                    C(1, 2 * l_index - 1 + phi_mod_2) = 0;
                    PM(l_index) = PM_pair(1, l_index);
                case 3
                    index = kill_index(kill_cnt);  %�����һ�� kill ��·��
                    kill_cnt = kill_cnt - 1;%pop stack
                    activepath(index) = 1;
                    %lazy copy
                    lazy_copy(:, index) = lazy_copy(:, l_index);
                    u(:, index) = u(:, l_index);
                    u(cnt_u, l_index) = 0;
                    u(cnt_u, index) = 1;
                    C(1, 2 * l_index - 1 + phi_mod_2) = 0;
                    C(1, 2 * index - 1 + phi_mod_2) = 1;
                    PM(l_index) = PM_pair(1, l_index);
                    PM(index) = PM_pair(2, l_index);
            end
        end
        cnt_u = cnt_u + 1;
    else%frozen bit operation
        for l_index = 1 : L
            if activepath(l_index) == 0
                continue;
            end
            if P(1, l_index) < 0
                PM(l_index) = PM(l_index) - P(1, l_index);
            end
            if phi_mod_2 == 0
                C(1, 2 * l_index - 1) = 0;
            else
                C(1, 2 * l_index) = 0;
            end 
        end
    end 
    
    for l_index = 1 : L%partial-sum return
        if activepath(l_index) == 0
            continue
        end
        if (phi_mod_2  == 1) && (phi ~= N - 1)
            layer = bit_layer_vec(phi + 1);
            for i_layer = 0 : layer - 1
                index_1 = lambda_offset(i_layer + 1);
                index_2 = lambda_offset(i_layer + 2);
                for beta = index_1 : 2 * index_1 - 1
                    C(beta + index_1, 2 * l_index) = mod(C(beta, 2 *  lazy_copy(i_layer + 1, l_index) - 1) + C(beta, 2 * l_index), 2);%Left Column lazy copy
                    C(beta + index_2, 2 * l_index) = C(beta, 2 * l_index);   
                end
            end
            index_1 = lambda_offset(layer + 1);
            index_2 = lambda_offset(layer + 2);
            for beta = index_1 : 2 * index_1 - 1
                C(beta + index_1, 2 * l_index - 1) = mod(C(beta, 2 * lazy_copy(layer + 1, l_index) - 1) + C(beta, 2 * l_index), 2);%Left Column lazy copy
                C(beta + index_2, 2 * l_index - 1) = C(beta, 2 * l_index);
            end 
        end
    end
    %lazy copy
    if phi < N - 1
        for i_layer = 1 : llr_layer_vec(phi + 2) + 1
            for l_index = 1 : L
                lazy_copy(i_layer, l_index) = l_index;
            end
        end
    end
end
%path selection.
[~, min_index] = min(PM); %������������е�PMֵ������
polar_info_esti=u(:,min_index);
end



function polar_info_esti = CASCL_decoder(llr, L, K, frozen_bits, ...
   det,lambda_offset, llr_layer_vec, bit_layer_vec)
%LLR-based SCL deocoder, a single function, no other sub-functions.
%Frequently calling sub-functions will derease the efficiency of MATLAB
%codes.
%const
N = length(llr);
m = log2(N);
%memory declared
lazy_copy = zeros(m, L);%If Lazy Copy is used, there is no data-copy in the decoding process. We only need to record where the data come from. Here,data refer to LLRs and partial sums.
%Lazy Copy is a relatively sophisticated operation for new learners of polar codes. If you do not understand such operation, you can directly copy data.
%If you can understand lazy copy and you just start learning polar codes
%for just fews days, you are very clever,
P = zeros(N - 1, L); %Channel llr is public-used, so N - 1 is enough.
C = zeros(N - 1, 2 * L);%I do not esitimate (x1, x2, ... , xN), so N - 1 is enough.
u = zeros(K, L);%unfrozen bits that polar codes carry, including crc bits.
PM = zeros(L, 1);%Path metrics
activepath = zeros(L, 1);%Indicate if the path is active. '1'��active; '0' otherwise.
cnt_u = 1;%information bit counter 
%initialize
activepath(1) = 1;
lazy_copy(:, 1) = 1;
%decoding starts
%default: in the case of path clone, the origianl path always corresponds to bit 0, while the new path bit 1.
for phi = 0 : N - 1
    layer = llr_layer_vec(phi + 1);
    phi_mod_2 = mod(phi, 2);
    for l_index = 1 : L
        if activepath(l_index) == 0
            continue;
        end
        switch phi%Decoding bits u_0 and u_N/2 needs channel LLR, so the decoding of them is separated from other bits. 
            case 0
                index_1 = lambda_offset(m);
                for beta = 0 : index_1 - 1
                    P(beta + index_1, l_index) = sign(llr(beta + 1)) * sign(llr(beta + index_1 + 1)) * min(abs(llr(beta + 1)), abs(llr(beta + index_1 + 1)));
                end
                for i_layer = m - 2 : -1 : 0
                    index_1 = lambda_offset(i_layer + 1);
                    index_2 = lambda_offset(i_layer + 2);
                    for beta = 0 : index_1 - 1
                        P(beta + index_1, l_index) = sign(P(beta + index_2, l_index)) *...
                            sign(P(beta + index_1 + index_2, l_index)) * min(abs(P(beta + index_2, l_index)), abs(P(beta + index_1 + index_2, l_index)));
                    end
                end
            case N/2
                index_1 = lambda_offset(m);
                for beta = 0 : index_1 - 1
                    x_tmp = C(beta + index_1, 2 * l_index - 1);
                    P(beta + index_1, l_index) = (1 - 2 * x_tmp) * llr(beta + 1) + llr(beta + 1 + index_1);
                end
                for i_layer = m - 2 : -1 : 0
                    index_1 = lambda_offset(i_layer + 1);
                    index_2 = lambda_offset(i_layer + 2);
                    for beta = 0 : index_1 - 1
                        P(beta + index_1, l_index) = sign(P(beta + index_2, l_index)) *...
                            sign(P(beta + index_1 + index_2, l_index)) * min(abs(P(beta + index_2, l_index)), abs(P(beta + index_1 + index_2, l_index)));
                    end
                end
            otherwise
                index_1 = lambda_offset(layer + 1);
                index_2 = lambda_offset(layer + 2);
                for beta = 0 : index_1 - 1
                    P(beta + index_1, l_index) = (1 - 2 * C(beta + index_1, 2 * l_index - 1)) * P(beta + index_2, lazy_copy(layer + 2, l_index)) +...
                        P(beta + index_1 + index_2, lazy_copy(layer + 2, l_index));
                end
                for i_layer = layer - 1 : -1 : 0
                    index_1 = lambda_offset(i_layer + 1);
                    index_2 = lambda_offset(i_layer + 2);
                    for beta = 0 : index_1 - 1
                        P(beta + index_1, l_index) = sign(P(beta + index_2, l_index)) *...
                            sign(P(beta + index_1 + index_2, l_index)) * min(abs(P(beta + index_2, l_index)),...
                            abs(P(beta + index_1 + index_2, l_index)));
                    end
                end
        end
    end
    if frozen_bits(phi + 1) == 0%if now we decode an unfrozen bit
        PM_pair = realmax * ones(2, L);
        for l_index = 1 : L
            if activepath(l_index) == 0
                continue;
            end
            if P(1, l_index) >= 0
                PM_pair(1, l_index) = PM(l_index);
                PM_pair(2, l_index) = PM(l_index) + P(1, l_index);
            else
                PM_pair(1, l_index) = PM(l_index) - P(1, l_index);
                PM_pair(2, l_index) = PM(l_index);
            end
        end
        middle = min(2 * sum(activepath), L);
        PM_sort = sort(PM_pair(:));
        PM_cv = PM_sort(middle);
        compare = PM_pair <= PM_cv; 
        kill_index = zeros(L, 1);%to record the index of the path that is killed
        kill_cnt = 0;%the total number of killed path
        %the above two variables consist of a stack
        for i = 1 : L
            if (compare(1, i) == 0)&&(compare(2, i) == 0)%which indicates that this path should be killed
                activepath(i) = 0;
                kill_cnt = kill_cnt + 1;%push stack
                kill_index(kill_cnt) = i;
            end
        end
        for l_index = 1 : L
            if activepath(l_index) == 0
                continue;
            end
            path_state = compare(1, l_index) * 2 + compare(2, l_index);
            switch path_state%path_state can equal to 0, but in this case we do no operation.
                case 1
                    u(cnt_u, l_index) = 1;
                    C(1, 2 * l_index - 1 + phi_mod_2) = 1;
                    PM(l_index) = PM_pair(2, l_index);
                case 2
                    u(cnt_u, l_index) = 0;
                    C(1, 2 * l_index - 1 + phi_mod_2) = 0;
                    PM(l_index) = PM_pair(1, l_index);
                case 3
                    index = kill_index(kill_cnt);
                    kill_cnt = kill_cnt - 1;%pop stack
                    activepath(index) = 1;
                    %lazy copy
                    lazy_copy(:, index) = lazy_copy(:, l_index);
                    u(:, index) = u(:, l_index);
                    u(cnt_u, l_index) = 0;
                    u(cnt_u, index) = 1;
                    C(1, 2 * l_index - 1 + phi_mod_2) = 0;
                    C(1, 2 * index - 1 + phi_mod_2) = 1;
                    PM(l_index) = PM_pair(1, l_index);
                    PM(index) = PM_pair(2, l_index);
            end
        end
        cnt_u = cnt_u + 1;
    else%frozen bit operation
        for l_index = 1 : L
            if activepath(l_index) == 0
                continue;
            end
            if P(1, l_index) < 0
                PM(l_index) = PM(l_index) - P(1, l_index);
            end
            if phi_mod_2 == 0
                C(1, 2 * l_index - 1) = 0;
            else
                C(1, 2 * l_index) = 0;
            end 
        end
    end 
    
    for l_index = 1 : L%partial-sum return
        if activepath(l_index) == 0
            continue
        end
        if (phi_mod_2  == 1) && (phi ~= N - 1)
            layer = bit_layer_vec(phi + 1);
            for i_layer = 0 : layer - 1
                index_1 = lambda_offset(i_layer + 1);
                index_2 = lambda_offset(i_layer + 2);
                for beta = index_1 : 2 * index_1 - 1
                    C(beta + index_1, 2 * l_index) = mod(C(beta, 2 *  lazy_copy(i_layer + 1, l_index) - 1) + C(beta, 2 * l_index), 2);%Left Column lazy copy
                    C(beta + index_2, 2 * l_index) = C(beta, 2 * l_index);   
                end
            end
            index_1 = lambda_offset(layer + 1);
            index_2 = lambda_offset(layer + 2);
            for beta = index_1 : 2 * index_1 - 1
                C(beta + index_1, 2 * l_index - 1) = mod(C(beta, 2 * lazy_copy(layer + 1, l_index) - 1) + C(beta, 2 * l_index), 2);%Left Column lazy copy
                C(beta + index_2, 2 * l_index - 1) = C(beta, 2 * l_index);
            end 
        end
    end
    %lazy copy
    if phi < N - 1
        for i_layer = 1 : llr_layer_vec(phi + 2) + 1
            for l_index = 1 : L
                lazy_copy(i_layer, l_index) = l_index;
            end
        end
    end
end
%path selection.
[~, path_ordered] = sort(PM); %������������е�PMֵ������
for l_index = 1 : L
    path_num = path_ordered(l_index);
    info_with_crc = u(:, path_num);
%     err = sum(mod(H_crc * info_with_crc, 2));
    [~,err]=detect(det,info_with_crc);
    if err == 0
        polar_info_esti = u(:, path_num);
        break;
    else
        if l_index == L
            polar_info_esti = u(:, path_ordered(1));
        end
    end
end 
end