% Mohammad Ismail Hossain 
% Jacobs University Bremen
% Waterfilling Alogorithm
clc
clear all;
Lambda=[5.0000    3.3333    1.2500];
Lambda_Inv = 1./Lambda;

Number_Channel= length(Lambda_Inv) ; 
[S_Number dt]=sort(Lambda_Inv);
sum(Lambda_Inv)
for p=length(S_Number):-1:1
    T_P=(1+sum(S_Number(1:p)))/p;
    Input_Power=T_P-S_Number;
    Pt=Input_Power(1:p);
    if(Pt(:)>=0),
        break
    end
end

% biselection, 二分法
% P should be close to Allocated_Power
% mu should be close to T_P
[P,mu] = water_filling(Lambda,1,1)

Allocated_Power=zeros(1,Number_Channel);
Allocated_Power(dt(1:p))=Pt;
Capacity=sum(log2(1+Allocated_Power./Lambda_Inv));
for ii =1:length(Lambda_Inv)
    g(ii,:)=[Lambda_Inv(ii),Allocated_Power(ii)];
end
bar(g,'stack');
legend ('Noise Level','Power Level')
ylabel('Noise & Power Level','fontsize',12)
xlabel('Number of Channels (N)','fontsize',12)
title('Power Allocation for Waterfilling Alogorithm','fontsize',12)




%% 注水算法函数
function [P, mu] = water_filling(gains, noise_var, P_total)
    N = length(gains);
    P = zeros(N, 1);
    mu = 0;
    low = 0;
    high = 1e6;
    tol = 1e-6;
    while (high - low) > tol
        mu = (low + high) / 2;
        P_tmp = max(mu - noise_var ./ gains, 0);
        if sum(P_tmp) <= P_total
            low = mu;
        else
            high = mu;
        end
    end
    mu = (low + high) / 2;
    P = max(mu - noise_var ./ gains, 0);
end