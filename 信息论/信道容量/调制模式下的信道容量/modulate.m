function [s,ref]=modulate(MOD_TYPE,M,d,COHERENCE)
%Wrapper function to call various digital modulation techniques
%  MOD_TYPE - 'PSK','QAM','PAM','FSK'
%  M - modulation order, For BPSK M=2, QPSK M=4, 256-QAM M=256 etc..,
%  d - data symbols to be modulated drawn from the set {1,2,...,M}
% COHERENCE - only applicable if FSK modulation is chosen
%           - 'coherent' for coherent MFSK
%           - 'noncoherent' for coherent MFSK
%  s - modulated symbols 
%  ref - ideal constellation points that could be used by an IQ detector
switch lower(MOD_TYPE)
    case 'bpsk'
        [s,ref] = mpsk_modulator(2,d);
    case 'psk'
        [s,ref] = mpsk_modulator(M,d);
    case 'qam'
        [s,ref] = mqam_modulator(M,d);
    case 'pam'
        [s,ref] = mpam_modulator(M,d);
    case 'fsk'
        [s,ref] = mfsk_modulator(M,d,COHERENCE);
    otherwise
        error('Invalid Modulation specified');
end