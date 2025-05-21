function r = eff_eval(p, w1, w2, d, index_SiH, index_SiO, n0)
addpath(genpath('../reticolo_allege_v9'))
h = 110;
% n0 = 1.48;
r = [];
lamda = index_SiH(:, 1);
for i = 1: length(lamda)% longueur d'onde
    retio;
    LD = lamda(i);
    
    n1 = index_SiH(i, 2) + 1i * index_SiH(i, 3);
    n2 = index_SiO(i, 2) + 1i * index_SiO(i, 3);
    
    D=[p, p];
    teta0 = 0;
    ro = n0 * sin(teta0 * pi / 180);
    
    delta0 = 0;
    parm = res0;
    parm.not_io = 1;
    parm.sym.x = 0;
    parm.sym.y = 0;
    parm.sym.pol = 1;

    nn = [8, 8];
    
    textures{1} = n0;   
    textures{2} = n2; 
    textures{3} = {n0, [0, 0, w1, w1, n1, 1], [0, 0, w1-2*d, w1-2*d, n0, 1], [0, 0, w2, w2, n1, 1]};
    aa = res1(LD, D, textures, nn, ro, delta0, parm);

    profil = {[0, h, 0], [1, 3, 2]};
    ef = res2(aa, profil);
    ref = ef.TEinc_top_reflected.efficiency_TE{0, 0};
    r = [r, ref];
    retio;
end

end

