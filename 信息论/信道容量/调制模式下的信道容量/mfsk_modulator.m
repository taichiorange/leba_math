function [s,ref]= mfsk_modulator(M,d,COHERENCE)
%Function to MFSK modulate the vector of data symbols - d
%[s,ref]=mfsk_modulator(M,d,COHERENCE) modulates the symbols defined 
%by the vector d using MFSK modulation, where M specifies the order 
%of M-FSK modulation and the vector d contains symbols whose values
%in the range 1:M. 
%The parameter 'COHERENCE' = 'COHERENT' or 'NONCOHERENT' specifies 
%the type of MFSK modulation/detection. The output s is the 
%modulated output and ref represents the reference constellation 
%that can be used during coherent demodulation.
if strcmpi(COHERENCE,'coherent'),
    phi= zeros(1,M); %phase=0 for coherent detection
    ref = complex(diag(exp(1i*phi)));%force complex data type 
    s = complex(ref(d,:)); %force complex type, since imag part is 0
else
    phi = 2*pi*rand(1,M);%M random phases in the (0,2pi)
    ref = diag(exp(1i*phi));
    s = ref(d,:);
end; end