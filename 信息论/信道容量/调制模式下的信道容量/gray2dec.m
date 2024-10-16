function [decimal]=gray2dec(grayInput)
%convert Gray code to decimal
%example: y=[0 1 3 2 6 7 5 4] %Gray coded
%x = gray2dec(y)
%returns x = [0 1 2 3 4 5 6 7] %decimal
grayInput=uint8(grayInput); %Force datatype to uint8
[rows,cols]=size(grayInput);
decimal=zeros(rows,cols);
for i=1:rows
 for j=1:cols
  temp = bitxor(grayInput(i,j),bitshift(grayInput(i,j),-8));
  temp = bitxor(temp,bitshift(temp,-4));
  temp = bitxor(temp,bitshift(temp,-2));
  temp = bitxor(temp,bitshift(temp,-1));
  decimal(i,j) = temp;
 end
end