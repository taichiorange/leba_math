clear;
close all;

A = [
1 1 1 0 0 1 1 0 0 1
1 0 1 0 1 1 0 1 1 0
0 0 1 1 1 0 1 0 1 1
0 1 0 1 1 1 0 1 0 1
1 1 0 1 0 0 1 1 1 0];


% First set the channel posterior probabilities
p1 = [.22  .16  .19  .48 .55  .87 .18 .79 .25 .76];

Nloop = 50;


% Do belief propagation on the Gallager code described by the
% parity check matrix A, using the prior probabilities in p1.
% At most Nloop decoding iterations are computed

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
r0 = zeros(M,N);
r1 = zeros(M,N);
q1 = zeros(M,N);
q0 = zeros(M,N);
alpha = zeros(M,N);
for m=1:M
  for n=1:N
    if(A(m,n)) 
        q1(m,n) = p1(n); q0(m,n) = 1-p1(n); 
    end
  end
end

q0pp = zeros(1,N);
q1pp = zeros(1,N);
for loop = 1:Nloop
    fprintf(1,'loop=%d\n',loop);
    deltaq = q0 - q1;
    deltar = zeros(M,N);
    for m = 1:M 			  % for each row
        for jj=1:length(Nl{m}) % work across the columns ("horizontally")
          n = Nl{m}(jj);
          pr = 1;
          for lp = 1:length(Nl{m})
            if(lp == jj) 
                continue; 
            end
            pr = pr*deltaq(m,Nl{m}(lp));
          end
          deltar(m,n) = pr;
        end
    end
    r0(idx) = (1+deltar(idx))/2;
    r1(idx) = (1-deltar(idx))/2;


    for n=1:N 			        % for each column
        for m1 = 1:length(Ml{n})	% work down the rows ("vertically")
            m = Ml{n}(m1);
            pr0pp = 1-p1(n);
            pr1pp = p1(n);
            pr0 = 1-p1(n);
            pr1 = p1(n);
            for mp=1:length(Ml{n})
                if(mp == m1)
                    r0save = r0(Ml{n}(mp),n);
                    r1save = r1(Ml{n}(mp),n);
                    continue;
                end
                pr0 = pr0*r0(Ml{n}(mp),n);
                pr1 = pr1*r1(Ml{n}(mp),n);
            end
            q0(m,n) = pr0;
            q1(m,n) = pr1;
        end
        q0pp(n) = pr0*r0save;
        q1pp(n) = pr1*r1save;
        alphapp = q0pp(n) + q1pp(n);
        q0pp(n) = q0pp(n)/alphapp;
        q1pp(n) = q1pp(n)/alphapp;
    end

    alpha(idx) = q0(idx) + q1(idx);
    q0(idx) = q0(idx) ./ alpha(idx);
    q1(idx) = q1(idx) ./ alpha(idx);


    x = q1pp > 0.5;
    z1 = mod(A*x',2);
    if(all(z1==0))
      break;
    end
% return
end  % end for loop

if(~all(z1==0))
    fprintf(1,'Decoding failure after %d iterations',Nloop);
end
display(x);