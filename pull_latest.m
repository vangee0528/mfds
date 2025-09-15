% 脚本：从GitHub拉取最新版本
% 假设当前目录是仓库根目录

repoPath = pwd;  % 获取当前工作目录

% 创建Git仓库对象
repo = matlab.git.GitRepository(repoPath);

% 执行pull操作，从origin拉取main分支
try
    repo.pull();
    disp('成功从 https://github.com/vangee0528/mfds.git 拉取最新版本。');
catch ME
    disp(['拉取失败：', ME.message]);
end
