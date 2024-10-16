function [varargout]= constructQAM(M)
%Function to construct gray codes symbol constellation for M-QAM
% [ref]=constructQAM(M) - returns the ideal signaling points (ref) 
% in a symmetric rectangular M-QAM constellation, where M is the 
% level of QAM modulation. The returned constellation points are 
% arranged such that the index of the points are arranged in a 
% Gray-coded manner. When plotted, indices of constellation points
% will differ by 1 bit.
%
% [ref,I,Q]=constructQAM(M) - returns the ideal signaling points  
% (ref) along with the IQ components breakup in a symmetric 
% rectangular M-QAM constellation, where M is the level of QAM 
% modulation. The returned constellation points are arranged such  
% that the index of the points are arranged in a Gray-coded manner. 
% When plotted, the indices of the  constellation points will 
% differ by 1 bit.
n=0:1:M-1; %Sequential address from 0 to M-1 (1xM dimension)

%------Addresses in Kmap - Gray code walk---------------
a=dec2gray(n); %Convert linear addresses to gray code
N=sqrt(M); %Dimension of K-Map - N x N matrix
a=reshape(a,N,N).'; %NxN gray coded matrix
evenRows=2:2:size(a,1); %identify alternate rows
a(evenRows,:)=fliplr(a(evenRows,:));%Flip rows - KMap representation
nGray=reshape(a.',1,M); %reshape to 1xM - Gray code walk on KMap

%Construction of ideal M-QAM constellation from sqrt(M)-PAM
D=sqrt(M); %Dimension of PAM constellation
x=floor(nGray/D);
y=mod(nGray,D);
Ax=2*(x+1)-1-D; %PAM Amplitudes 2m-1-D - real axis
Ay=2*(y+1)-1-D; %PAM Amplitudes 2m-1-D - imag axis
varargout{1}=Ax+1i*Ay; %M-QAM points (I+jQ)
varargout{2}=Ax; %Real part (I)
varargout{3}=Ay; %Imaginary part (Q)
