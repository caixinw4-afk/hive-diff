%% gen_tonemap.m
% Tone-map every scene folder of HDR frames (EXR / HDR / TIFF) under IN_ROOT to 8-bit PNG with
% MATLAB's built-in tonemap, mirroring the folder structure to OUT_ROOT:
%   IN_ROOT/<scene>/*.exr  ->  OUT_ROOT/<scene>/*.png
% The frames of a scene are first normalised with one global pair of percentiles (median over
% frames of the per-frame P_LO / P_HI percentiles) so the tone-mapping does not flicker.

IN_ROOT  = '/path/to/HdM-HDR-2014_Original-HDR-Camera-Footage';
OUT_ROOT = '/path/to/HdM-HDR-2014_tonemapped';

EXTS       = {'.exr','.hdr','.tif','.tiff'};
OUT_FORMAT = '.png';

P_LO = 0.5;     % low percentile (%)
P_HI = 99.5;    % high percentile (%)

ADJUST_SAT = 3;        % tonemap AdjustSaturation
NUM_TILES  = [4 4];    % tonemap NumberOfTiles

if ~exist(OUT_ROOT,'dir'), mkdir(OUT_ROOT); end
scenes = dir(IN_ROOT);
scenes = scenes([scenes.isdir] & ~ismember({scenes.name},{'.','..'}));
fprintf('[INFO] %d scene folder(s) found under %s\n', numel(scenes), IN_ROOT);

for s = 1:numel(scenes)
    scene_name = scenes(s).name;
    in_scene   = fullfile(IN_ROOT,  scene_name);
    out_scene  = fullfile(OUT_ROOT, scene_name);
    if ~exist(out_scene,'dir'), mkdir(out_scene); end
    fprintf('\n[SCENE] %s\n', scene_name);

    files = [];
    for k = 1:numel(EXTS)
        files = [files; dir(fullfile(in_scene, ['*' EXTS{k}]))]; %#ok<AGROW>
    end
    if isempty(files)
        fprintf('  [WARN] no HDR frames, skip.\n');
        continue;
    end
    [~,ord] = sort({files.name});
    files = files(ord);

    % pass 1: global normalisation range
    lo_list = zeros(numel(files),1);
    hi_list = zeros(numel(files),1);
    for i = 1:numel(files)
        X = read_hdr_any(fullfile(files(i).folder, files(i).name));
        if size(X,3) == 1, X = repmat(X,[1 1 3]); end
        xi = X(:);
        lo_list(i) = prctile(xi, P_LO);
        hi_list(i) = prctile(xi, P_HI);
    end
    g_lo = median(lo_list);
    g_hi = median(hi_list);
    if ~isfinite(g_lo) || ~isfinite(g_hi) || g_hi <= g_lo
        g_lo = min(lo_list); g_hi = max(hi_list);
    end
    fprintf('  [INFO] global p%g=%.6g, p%g=%.6g\n', P_LO, g_lo, P_HI, g_hi);

    % pass 2: normalise, tone-map, write
    for i = 1:numel(files)
        in_path = fullfile(files(i).folder, files(i).name);
        [~,stem,~] = fileparts(files(i).name);
        out_path = fullfile(out_scene, [stem OUT_FORMAT]);
        try
            X = read_hdr_any(in_path);
            if size(X,3) == 1, X = repmat(X,[1 1 3]); end
            Xn = (X - g_lo) / max(g_hi - g_lo, eps('single'));
            Xn = min(max(Xn,0),1);
            RGB = tonemap(Xn, AdjustSaturation = ADJUST_SAT, NumberOfTiles = NUM_TILES);
            imwrite(im2uint8(RGB), out_path);
            if mod(i,25) == 0 || i == numel(files)
                fprintf('    [%4d/%4d] %s -> %s\n', i, numel(files), files(i).name, out_path);
            end
        catch ME
            fprintf('    [ERR] %s : %s\n', in_path, ME.message);
        end
    end
    fprintf('  [DONE] %s -> %s\n', in_scene, out_scene);
end
fprintf('\n[ALL DONE]\n');

function X = read_hdr_any(p)
    [~,~,ext] = fileparts(lower(p));
    switch ext
        case '.exr'
            if exist('exrread','file'), X = exrread(p); else, X = imread(p); end
        case {'.hdr','.pic'}
            if exist('hdrread','file'), X = hdrread(p); else, X = imread(p); end
        otherwise
            X = imread(p);
    end
    X = im2single(X);
end
