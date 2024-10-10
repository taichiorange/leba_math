% Demonstrate LDPC decoding

% Copyright 2004 by Todd K. Moon
% Permission is granted to use this program/data
% for educational/research only

clear;

% A used in chapter
A = [
1 1 1 0 0 1 1 0 0 1
1 0 1 0 1 1 0 1 1 0
0 0 1 1 1 0 1 0 1 1
0 1 0 1 1 1 0 1 0 1
1 1 0 1 0 0 1 1 1 0];

a = 2;   % signal amplitude
sigma2 = 2;  % noise variance

% First set the channel posterior probabilities
p1 = [.22  .16  .19  .48 .55  .87 .18 .79 .25 .76]; 

% then compute the received values that correspond to these
r =  log((1./p1)-1)/(-2*a)*sigma2;  % received vector

Nloop = 50;

Lc = 2*a/sigma2;


%%%% to do decode by log likelihood ratio
% x : decoded, should be [  0   0   0   1   0   1   0   1   0   1 ]


[M,N] = size(A);
clear Nl Ml
Nl = cell(M);
Ml = cell(N);
for m=1:M 
    Nl{m} = []; 
end
for n=1:N 
    Ml{n} = []; 
end
% Build the sparse representation of A using the M and N sets
for m=1:M
  for n=1:N
	if(A(m,n))
	  Nl{m} = [Nl{m} n];
	  Ml{n} = [Ml{n} m];
	end
  end
end
idx = find(A ~= 0);  % identify the "sparse" locations of A
% The index vector idx is used to emulate the sparse operations
% of the decoding algorithm.  In practice, true sparse operations
% and spare storage should be used.

% Initialize the probabilities
eta = zeros(M,N);
lasteta = zeros(M,N);
lambda = Lc*r;

for loop = 1:Nloop
  fprintf(1,'loop=%d\n',loop);
  for m = 1:M 			  % for each row (check)
	for n=Nl{m} % work across the columns ("horizontally")
	  pr = 1;
	  for np= Nl{m}
        if np == n 
            continue; 
        end
		pr = pr*tanh((-lambda(np)+lasteta(m,np))/2); % accumulate the product
	  end
	  eta(m,n) = -2*atanh(pr);
	end
  end
  lasteta = eta;   % save to subtract to obtain extrinsic for next time around

  for n=1:N 			        % for each column (bit)
	lambda(n) = Lc*r(n);
	for m = Ml{n}	% work down the rows ("vertically")
	  lambda(n) = lambda(n) + eta(m,n);
	end
  end

  x = lambda >= 0;  % compute decoded bits for stopping criterion
  z1 = mod(A*x',2)';

  if(all(z1==0)) 
      break; 
  end
end  % end for loop

if(~all(z1==0))
  fprintf(1,'Decoding failure after %d iterations',Nloop);
end