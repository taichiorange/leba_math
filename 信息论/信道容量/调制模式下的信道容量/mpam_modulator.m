function [s,ref]=mpam_modulator(M,d)
%Function to MPAM modulate the vector of data symbols - d
%[s,ref]=mpam_modulator(M,d) modulates the symbols defined by the
%vector d using MPAM modulation, where M specifies the order of 
%M-PAM modulation and the vector d contains symbols whose values 
%in the range 1:M. The output s is the modulated output and ref 
%represents the reference constellation
m=1:1:M; 
Am=complex(2*m-1-M); %All possibe amplitude levels
s = complex(Am(d)); %M-PAM transmission
ref = Am; %reference constellation    
end