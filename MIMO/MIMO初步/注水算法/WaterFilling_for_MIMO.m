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