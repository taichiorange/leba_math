clear;

i=1;
dT = 0.001;
Lambda=[5.0000    3.3333    1.2500];
for x=0:dT:1
    j=1;
    for y=0:dT:1
       if y>(1-x)
		z=0;
	else
		z = log2(1+Lambda(1)*x) + log2(1+Lambda(2)*y) + log2(1+Lambda(3)*(1-x-y));
	end
	Z(i,j) = z;
	j=j+1;
    end
    i=i+1;
end

X = 0:dT:1;
Y=0:dT:1;

[M1,I] = max(Z,[],1);
[M,J] = max(M1);
Px=X(J)
Py = X(I(J))
Pz = 1- Px - Py
M

figure;
mesh(X,Y,Z);