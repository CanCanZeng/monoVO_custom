%% read input image sequence and ground truth

images = imageDatastore(fullfile(toolboxdir('vision'), 'visiondata', ...
    'NewTsukuba'));

% create the camera parameters object using camera intrinsics from the
% New Tsukuba dataset.
K = [615 0 320; 0 615 240; 0 0 1]';
cameraParams = cameraParameters('IntrinsicMatrix', K);

% load ground truth camera poses.
load(fullfile(toolboxdir('vision'), 'visiondata', ...
    'visualOdometryGroundTruth.mat'));

%% create a view set containing the first view of the sequence

% create an empty viewSet object to manage the data associated with
% each view.
vSet = viewSet;

% read and display the first image.
Irgb = readimage(images, 1);
player = vision.VideoPlayer('Position', [20,400,650,510]);
step(player, Irgb);

prevI = undistortImage(rgb2gray(Irgb), cameraParams);

% detect features.
prevPoints = detectSURFFeatures(prevI, 'MetricThreshold', 500);

% select a subset of features, uniformly distributed throughout the image.
numPoints = 150;
prevPoints = selectUniform(prevPoints, numPoints, size(prevI));

% extract features. using 'Upright' features improves matching quality
% if the camera motion involves little or no in-plane rotation.
prevFeatures = extractFeatures(prevI, prevPoints, 'Upright', true);

% 添加第一个视角。把第一个相机放在世界坐标系的原点，并让
% 它的方向朝向Z轴正方向
viewId = 1;
vSet = addView(vSet, viewId, 'Points', prevPoints, 'Orientation', eye(3),...
    'Location', [0,0,0]);

%% 显示初始的相机位置

% 设置坐标轴
figure
axis([-220, 50, -140, 20, -50, 300]);

% 设置Y轴竖直向下
view(gca, 3);
set(gca, 'CameraUpVector', [0,-1,0]);
camorbit(gca, -120, 0, 'data', [0,1,0]);

grid on
xlabel('X (cm)');
ylabel('Y (cm)');
zlabel('Z (cm)');
hold on

% 画出估计出来的相机位姿
cameraSize = 7;
camEstimated = plotCamera('Size', cameraSize, 'Location',...
    vSet.Views.Location{1}, 'Orientation', vSet.Views.Orientation{1},...
    'Color', 'g', 'Opacity', 0);

% 画出实际相机位姿
camActual = plotCamera('Size', cameraSize, 'Location',...
    groundTruthPoses.Location{1}, 'Orientation',...
    groundTruthPoses.Orientation{1}, 'Color','b','Opacity',0);

% 初始化相机的轨迹
trajectoryEstimated = plot3(0,0,0,'g-');
trajectoryActual       = plot3(0,0,0,'b-');

legend('Estimated Trajectory', 'Actual Trajectory');
title('Camera Trajectory');

%% 估计第二个视角的位姿

% 读取和显示图片
viewId = 2;
Irgb = readimage(images, viewId);
step(player, Irgb);

% 将图片转换成灰度图并去畸变
I = undistortImage(rgb2gray(Irgb), cameraParams);

% 匹配当前图片特征和前一时刻的特征
[currPoints, currFeatures, indexPairs] = helperDetectAndMatchFeatures(...
    prevFeatures, I);

% 估计当前视角和前一视角之间的相对位姿
[orient, loc, inlierIdx] = helperEstimateRelativePose(...
    prevPoints(indexPairs(:,1)), currPoints(indexPairs(:,2)),...
    cameraParams);

% 去掉不满足极线约束的 outlier
indexPairs = indexPairs(inlierIdx, :);

% 添加当前视角到 view set 中
vSet = addView(vSet, viewId, 'Points', currPoints, 'Orientation',...
    orient, 'Location', loc);
% 存储当前帧和前一帧之间的特征点的匹配关系
vSet = addConnection(vSet, viewId-1, viewId, 'Matches', indexPairs);

vSet = helperNormalizeViewSet(vSet, groundTruthPoses);
helperUpdateCameraPlots(viewId, camEstimated, camActual,...
    poses(vSet), groundTruthPoses);
helperUpdateCameraTrajectories(viewId, trajectoryEstimated, trajectoryActual,...
    poses(vSet), groundTruthPoses);

prevI = I;
prevFeatures = currFeatures;
prevPoints = currPoints;

%% 前端估计相机的位姿，用全局BA

for viewId = 3:  15
    % 读取和显示下一帧图片
    Irgb = readimage(images, viewId);
    step(player, Irgb);
    
    % 转换成灰度图并去畸变
    I = undistortImage(rgb2gray(Irgb), cameraParams);
    
    % 匹配当前帧和前一帧的特征点
    [currPoints, currFeatures, indexPairs] = helperDetectAndMatchFeatures(...
        prevFeatures, I);
    
    % 去除 outliers
    inlierIdx = helperFindEpipolarInliers(prevPoints(indexPairs(:,1)),...
    currPoints(indexPairs(:,2)), cameraParams);
    
    % 三角化前两帧里面匹配好的特征点，并找到当前帧中对应的点
    [worldPoints, imagePoints] = helperFind3Dto2DCorrespondences(vSet,...
        cameraParams, indexPairs, currPoints);
    
    % 关闭 RANSAC 达到最大迭代次数的警报
    warningstate = warning('off', 'vision:ransac:maxTrialsReached');

    % 估计当前帧在世界坐标系下的位姿
    [orient, loc] = estimateWorldCameraPose(imagePoints, ...
        worldPoints, cameraParams, 'Confidence', 99.99, ...
        'MaxReprojectionError', 0.8);
    
    % 重新恢复报警
    warning(warningstate);
    
    % 添加当前帧到 view set 中
    vSet = addView(vSet, viewId, 'Points', currPoints, 'Orientation',...
        orient, 'Location', loc);
    
    % 存储当前帧和前一帧之间的匹配关系
    vSet = addConnection(vSet, viewId-1, viewId, 'Matches', indexPairs);
    
    tracks = findTracks(vSet); % 跨过多帧图像寻找跟踪到的点
    
    camPoses = poses(vSet);  % 得到所有帧的位姿
    
    % 三角化这些 3D 点的初始位置
    xyzPoints = triangulateMultiview(tracks, camPoses, cameraParams);
    
    % 使用 BA 优化相机的位姿
    [~, camPoses] = bundleAdjustment(xyzPoints, tracks, camPoses,...
        cameraParams, 'PointsUndistorted', true, 'AbsoluteTolerance',...
        1e-9, 'RelativeTolerance', 1e-9, 'MaxIterations', 300);
    
    vSet = updateView(vSet, camPoses); % 更新 view set.
    
    % BA 可能会移动所有的位姿，所以需要调整第一帧的方向，让其...
    % 朝向Z轴正方向，并且调整尺度
    vSet = helperNormalizeViewSet(vSet, groundTruthPoses);
    
    % 更新相机的轨迹
    helperUpdateCameraPlots(viewId, camEstimated, camActual,...
        poses(vSet), groundTruthPoses);
    helperUpdateCameraTrajectories(viewId, trajectoryEstimated,...
        trajectoryActual, poses(vSet), groundTruthPoses);
    
    prevI = I;
    prevFeatures = currFeatures;
    prevPoints     = currPoints;
end

for viewId = 16 : numel(images.Files)
    Irgb = readimage(images, viewId);
    step(player, Irgb);
    
    I = undistortImage(rgb2gray(Irgb), cameraParams);
    
    [currPoints, currFeatures, indexPairs] = helperDetectAndMatchFeatures(...
        prevFeatures, I);
    
    [worldPoints, imagePoints] = helperFind3Dto2DCorrespondences(vSet,...
        cameraParams, indexPairs, currPoints);
    
    warningstate = warning('off', 'vision:ransac:maxTrialsReached');
    
    [orient, loc] = estimateWorldCameraPose(imagePoints, worldPoints, ...
        cameraParams, 'MaxNumTrials', 5000, 'Confidence', 99.99, ...
        'MaxReprojectionError', 0.8);
    
    warning(warningstate);
    
    vSet = addView(vSet, viewId, 'Points', currPoints, 'Orientation', orient,...
        'Location', loc);
    vSet = addConnection(vSet, viewId-1, viewId, 'Matches', indexPairs);
    
    % 前面都是一般流程,只有这里,因为每次都做优化太慢了,所以每隔帧优化一次
    if mod(viewId, 7) ==0
        % 只在最近的 15 帧里面找可以跟踪到的点
        windowSize = 15;
        startFrame = max(1, viewId - windowSize);
        tracks = findTracks(vSet, startFrame:viewId);
        camPoses = poses(vSet, startFrame:viewId);
        [xyzPoints, reprojErrors] = triangulateMultiview(tracks, camPoses,...
            cameraParams);
        
        % 令前两帧固定,以保持尺度不变
        fixedIds = [startFrame, startFrame+1];
        
        % 把那些重投影误差比较大的都排除掉
        idx = reprojErrors < 2;
   
        [~, camPoses] = bundleAdjustment(xyzPoints(idx, :), tracks(idx),...
            camPoses, cameraParams, 'FixedViewIDs', fixedIds, ...
            'PointsUndistorted', true, 'AbsoluteTolerance', 1e-9,...
            'RelativeTolerance', 1e-9, 'MaxIterations', 300);
        
        vSet = updateView(vSet, camPoses);
    end
    
    helperUpdateCameraPlots(viewId, camEstimated, camActual, poses(vSet),...
        groundTruthPoses);
    helperUpdateCameraTrajectories(viewId, trajectoryEstimated, ...
        trajectoryActual, poses(vSet), groundTruthPoses);
    
    prevI = I;
    prevFeatures = currFeatures;
    prevPoints     = currPoints;
end

hold off









