function [grayCoded]=dec2gray(decInput)
%convert decimal to Gray code representation
%example: x= [0 1 2 3 4 5 6 7] %decimal
%y=dec2gray(x)
%returns y = [ 0 1 3 2 6 7 5 4] %Gray coded
[rows,cols]=size(decInput);
grayCoded=zeros(rows,cols);
for i=1:rows
 for j=1:cols
   grayCoded(i,j)=bitxor(bitshift(decInput(i,j),-1),decInput(i,j));
 end
end
