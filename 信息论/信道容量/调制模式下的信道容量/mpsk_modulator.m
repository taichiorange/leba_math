function [s,ref]=mpsk_modulator(M,d)
%Function to MPSK modulate the vector of data symbols - d
%[s,ref]=mpsk_modulator(M,d) modulates the symbols defined by the
%vector d using MPSK modulation, where M specifies the order of 
%M-PSK modulation and the vector d contains symbols whose values
%in the range 1:M. The output s is the modulated output and ref
%represents the reference constellation that can be used in demod
ref_i= 1/sqrt(2)*cos(((1:1:M)-1)/M*2*pi); 
ref_q= 1/sqrt(2)*sin(((1:1:M)-1)/M*2*pi);
ref = ref_i+1i*ref_q;
s = ref(d); %M-PSK Mapping
end