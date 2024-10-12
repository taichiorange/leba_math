clear;

Nstr = 2;  %  number of stream
Ntx = 4;  %number of tx antenna
Nsc = 1; % number of subcarries

bphi = 6;
bpsi = 4;

V = randn(Nsc,Nstr,Ntx) + 1j * randn(Nsc,Nstr,Ntx);

angidx = bfCompressQuantize(V,bphi, bpsi);

function angidx = bfCompressQuantize(VTilda,bphi,bpsi)
%bfCompressQuantize compresses the beamforming feedback matrix
%   ANGIDX = bfCompressQuantize(V,BPHI,BPSI) compresses the beamforming
%   feedback matrix, V into angles. There are two kinds of angles: phi and
%   psi. Those angles are quantized according to the bit resolution given
%   by BPHI and BPSI for phi and psi respectively. See Table 9-68
%   (Quantization of angles) in [1]. The size of V should be in the form of
%   (Number of active subcarriers)X(Nsts)X(Ntx).
%
%   References:
%   1) IEEE Standard for Information technology--Telecommunications and
%   information exchange between systems Local and metropolitan area
%   networks--Specific requirements - Part 11: Wireless LAN Medium Access
%   Control (MAC) and Physical Layer (PHY) Specifications," in IEEE Std
%   802.11-2016 (Revision of IEEE Std 802.11-2012) , vol., no., pp.1-3534,
%   Dec. 7 2016.

%   Copyright 2018 The MathWorks, Inc.

VTilda = permute(VTilda,[3 2 1]);
[Nr,Nc,Nst] = size(VTilda);
NumAngles = 2*max(sum(Nr-1:-1:max(Nr-Nc,1)),1);
angles = zeros(NumAngles,1,Nst);
angcnt = 0; % Angles count
% Calculate Dt (DTilda)
Dt = exp(1j*angle(conj(VTilda(end,:,:))));
v = VTilda.*Dt; % To make last row non-negative real.
v1 = v;
for ii = 1:min(Nc,Nr-1)
    phi = wrapTo2Pi(angle(v1(ii:end-1,ii,:))); % Eq 19-80 in [1]
    angles(angcnt+1:angcnt+size(phi,1),1,:) = phi;
    angcnt = angcnt + size(phi,1);
    Dii = [ones(ii-1,1,Nst); exp(1j*phi); ones(1,1,Nst)];
    v1 = conj(Dii).*v1;
    for ll = ii+1:Nr
        psi = atan(real(v1(ll,ii,:))./real(v1(ii,ii,:)));
        angles(angcnt+1,1,:) = psi;
        angcnt = angcnt + 1;
        for sc = 1: Nst % Find Givens rotation matrix for each subcarrier.
            G = eye(Nr);
            G(ii,ii) = cos(psi(:,:,sc));
            G(ii,ll) = sin(psi(:,:,sc));
            G(ll,ii) = -sin(psi(:,:,sc));
            G(ll,ll) = cos(psi(:,:,sc)); % Eq 19-81 in [1].
            v1(:,:,sc) = G(:,:)*v1(:,:,sc); % Eq 19-84 in [1].
        end
    end
end

%Quantization of all angles. See Table 9-68 (Quantization of angles) in [1]
angidx = zeros(Nst,NumAngles);
angcnt = 1;
for ii = Nr-1:-1:max(Nr-Nc,1)
    for jj = 1:ii
        angidx(:,angcnt) = round(0.5*((angles(angcnt,1,:)*((2^bphi))/pi)-1));
        angcnt = angcnt + 1;
    end
    
    for jj = 1:ii
        angidx(:,angcnt) = round(0.5*((angles(angcnt,1,:)*(2^(bpsi+2))/pi)-1));
        angcnt = angcnt + 1;
    end
end
end