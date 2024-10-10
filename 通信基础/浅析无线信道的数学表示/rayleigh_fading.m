
c = rayleighfading(10)

%Rayleigh fading
function [c]=rayleighfading(m)
	clc;
	N=32; % Number of reflections
	fmax=30; %Max doppler shift
	A=1; %amplitude
	f=10000; %sampling frequency
	t=0:1/f:((m/10000)-(1/f)); %sampling time
	ct=zeros(1,m);
	phi=2*pi* rand(1,32);    %公式9中相位，均匀分布
	theta=2*pi*rand(1,32);   % 公式9中，各个散射路径的入射角，均匀分布
	fd=fmax*cos(theta); %doppler shift，每个散射路径的多普勒频移
	for n=1:m       %时间
		for i=1:32    % 32 个散射路径，汇聚成一个不可分的单径
			ct(n)=ct(n)+(A*exp(j*(2*pi*fd(i)*t(n)+phi(i))));
		end
	end
	c=ct/sqrt(N); %channel coefficient	
end