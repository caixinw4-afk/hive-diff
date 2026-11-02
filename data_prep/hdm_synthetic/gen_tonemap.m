%% batch_tonemap_exr_run.m
% 递归遍历 IN_ROOT 下的子目录，查找 EXR/HDR/TIF，
% 用 MATLAB 内置 tonemap 生成 8-bit LDR，并按相同目录结构
% 保存到 OUT_ROOT 下。
% —— 直接点“Run”即可；只需改下面两个路径和可选参数。

%% ====== 必填：输入/输出根目录 ======
IN_ROOT  = '/path/to/data/hdm_moving_camera_rest';             % 每个子文件夹是一段视频
OUT_ROOT = '/path/to/data/hdm_moving_camera_rest_tone';    % 输出根目录（自动镜像子目录）
%% ====== 参数 ======
EXTS       = {'.exr','.hdr','.tif','.tiff'};    % 要处理的后缀
OUT_FORMAT = '.png';                             % 输出格式
RECURSE_IN_SCENE = false;                        % 场景里是否递归找子目录

P_LO = 0.5;    % 全局低分位（%）
P_HI = 99.5;   % 全局高分位（%）

% MATLAB 内置 tonemap 的参数
ADJUST_SAT   = 3;
%ADJUST_LIGHT = [0.01 0.99];
NUM_TILES    = [4 4];   % 若仍有轻微闪烁，可试 [2 2]

%% ========== 下面不用改 ==========
if ~exist(OUT_ROOT,'dir'), mkdir(OUT_ROOT); end
scenes = dir(IN_ROOT);
scenes = scenes([scenes.isdir] & ~ismember({scenes.name},{'.','..'}));
fprintf('[INFO] %d scene folder(s) found under %s\n', numel(scenes), IN_ROOT);

for s = 1:numel(scenes)
    scene_name = scenes(s).name;
    in_scene   = fullfile(IN_ROOT,  scene_name, "hdr_shaken_exr");
    out_scene  = fullfile(OUT_ROOT, scene_name, "tonemapped_png");
    if ~exist(out_scene,'dir'), mkdir(out_scene); end
    fprintf('\n[SCENE] %s\n', scene_name);

    % ---- 收集该场景的帧文件 ----
    files = [];
    if RECURSE_IN_SCENE
        for k=1:numel(EXTS)
            files = [files; dir(fullfile(in_scene,'**',['*' EXTS{k}]))]; %#ok<AGROW>
        end
    else
        for k=1:numel(EXTS)
            files = [files; dir(fullfile(in_scene, ['*' EXTS{k}]))]; %#ok<AGROW>
        end
    end
    if isempty(files)
        fprintf('  [WARN] no HDR frames, skip.\n'); 
        continue;
    end
    % 按名字排序（常见帧编号）
    [~,ord] = sort({files.name});
    files = files(ord);

    % ---- 第 1 遍：统计 p1 / p98（稳健近似：逐帧算再取中位数）----
    p1_list  = zeros(numel(files),1);
    p98_list = zeros(numel(files),1);
    for i=1:numel(files)
        X = read_hdr_any(fullfile(files(i).folder, files(i).name));
        if size(X,3)==1, X=repmat(X,[1 1 3]); end
        xi = X(:);
        % 可选：加快速度可抽样：xi = xi(1:10:end);
        p1_list(i)  = prctile(xi,  P_LO);
        p98_list(i) = prctile(xi, P_HI);
    end
    g_lo = median(p1_list);
    g_hi = median(p98_list);
    if ~isfinite(g_lo) || ~isfinite(g_hi) || g_hi <= g_lo
        g_lo = min(p1_list); g_hi = max(p98_list);
    end
    fprintf('  [INFO] global p%g=%.6g, p%g=%.6g\n', P_LO, g_lo, P_HI, g_hi);

    % ---- 第 2 遍：归一化 + tonemap + 保存 ----
    for i=1:numel(files)
        in_path = fullfile(files(i).folder, files(i).name);
        [~,stem,~] = fileparts(files(i).name);

        % 如果场景内递归，镜像子目录结构
        if RECURSE_IN_SCENE
            rel = erase(files(i).folder, in_scene);
            if ~isempty(rel) && rel(1)==filesep, rel = rel(2:end); end
            out_dir = fullfile(out_scene, rel);
        else
            out_dir = out_scene;
        end
        if ~exist(out_dir,'dir'), mkdir(out_dir); end
        out_path = fullfile(out_dir, [stem OUT_FORMAT]);

        try
            X = read_hdr_any(in_path);

            if size(X,3)==1, X=repmat(X,[1 1 3]); end

            Xn = (X - g_lo) / max(g_hi - g_lo, eps('single'));
            Xn = min(max(Xn,0),1);

            RGB = tonemap(Xn, ...
                AdjustSaturation = ADJUST_SAT, ...
                NumberOfTiles    = NUM_TILES);
%                AdjustLightness  = ADJUST_LIGHT, ...
            imwrite(im2uint8(RGB), out_path);
            if mod(i,25)==0 || i==numel(files)
                fprintf('    [%4d/%4d] %s -> %s\n', i, numel(files), files(i).name, out_path);
            end
        catch ME
            fprintf('    [ERR] %s : %s\n', in_path, ME.message);
        end
    end
    fprintf('  [DONE] %s -> %s\n', in_scene, out_scene);
end
fprintf('\n[ALL DONE]\n');

%% ------- 本脚本用到的本地函数 -------
function X = read_hdr_any(p)
    [~,~,ext] = fileparts(lower(p));
    switch ext
        case '.exr'
            if exist('exrread','file')
                X = exrread(p);
            else
                X = imread(p);     % 新版 MATLAB 通常支持
            end
        case {'.hdr','.pic'}
            if exist('hdrread','file')
                X = hdrread(p);
            else
                X = imread(p);
            end
        otherwise
            X = imread(p);         % 例如 16-bit TIFF/TIF
    end
    X = im2single(X);
end