// 全局用户状态
let globalUser = {
  id: null,         // 用户ID
  token: null,      // JWT或Access Token
  refreshToken: null, // 刷新令牌
  username: "",     // 用户名（可选）
  type: 0,        // 用户类型（0: 普通用户, 1: 管理员）
  school_info: "",  // 学校信息
  avatar_url: "", // 头像URL
  post_count: 0, // 帖子数量
  following_count: 0, // 关注数量
  follower_count: 0, // 粉丝数量
};

//---------------------------------------------------------------------------
// 在 DOM 加载完成后执行代码
document.addEventListener('DOMContentLoaded', function() {
    const postList = document.getElementById('post-list');
    const loadMoreBtn = document.getElementById('load-more');
    const loadIndicator = document.getElementById('loading-indicator');  // 加载指示器
    const noPosts = document.getElementById('no-posts');  // 无帖子提示
    const postModal = document.getElementById('post-modal');
    const postForm = document.getElementById('post-form');
    const imageUploadContainer = document.getElementById('image-upload-container');

    let currentPage = 1;  // 当前页码
    let isLoading = false;  // 是否正在加载数据
    let hasMorePosts = true;  // 是否还有更多帖子

    const followModal = document.getElementById('follow-modal');
    const followLink = document.getElementById('follow-link'); // 左侧边栏的"关注"链接
    const closeFollowModal = document.getElementById('close-follow-modal');
    const followList = document.getElementById('follow-list');
    const noFollow = document.getElementById('no-follow');
    const followLoading = document.getElementById('follow-loading');

    const likeModal = document.getElementById('like-modal');
    const likeLink = document.getElementById("like-link") // 左侧边栏的"点赞"链接
    const closeLikeModal = document.getElementById('close-like-modal');
    const likeList = document.getElementById('like-list');
    const noLike = document.getElementById('no-like');
    const likeLoading = document.getElementById('like-loading');

    const collectLink = document.getElementById("collect-link"); // 左侧边栏的"收藏"链接
    const collectModal = document.getElementById('collect-modal');
    const closeCollectModal = document.getElementById('close-collect-modal');
    const collectList = document.getElementById('collect-list');
    const noCollect = document.getElementById('no-collect');
    const collectLoading = document.getElementById('collect-loading');

    const recommendUsersContainer = document.getElementById('recommend-users');
    const recommendLoading = document.getElementById('recommend-loading');
    const noRecommend = document.getElementById('no-recommend');

    // 获取登录注册按钮和模态框元素
    const loginRegisterBtn = document.getElementById('login-register-btn');
    const loginRegisterModal = document.getElementById('login-register-modal');
    // 获取登录和注册选项卡及表单元素
    const loginTab = document.getElementById('loginTab');
    const registerTab = document.getElementById('registerTab');
    const loginPanel = document.getElementById('loginPanel');
    const registerPanel = document.getElementById('registerPanel');

    // 当前视图状态
    let currentView = 'all'; // all: 全站, followed: 关注
    const allPostsBtn = document.getElementById('all-posts-btn');
    const followedPostsBtn = document.getElementById('followed-posts-btn');
    
    allPostsBtn.addEventListener('click', function() {
        if (currentView === 'all') return;
        currentView = 'all';
        updateViewButtons();
        currentPage = 1;
        loadPosts(); 
    });

    // 绑定关注按钮和模态框
    followedPostsBtn.addEventListener('click', function() {
        if (currentView === 'followed') return;
        if (checkLoginStatusAndPrompt()) return; // 检查登录状态
        currentView = 'followed';
        updateViewButtons();
        currentPage = 1;
        loadPosts();
    });
    followLink.addEventListener('click', openFollowModal);
    if (closeFollowModal) {
        closeFollowModal.addEventListener('click', closeFollowModalFunc);
    }
    followModal.addEventListener('click', function(e) {
        if (e.target === followModal) {
            closeFollowModalFunc();
        }
    });

    // 绑定点赞模态框
    likeLink.addEventListener('click', openLikeModal); 
    if (closeLikeModal) {
        closeLikeModal.addEventListener('click', closeLikeModalFunc);
    }
    likeModal.addEventListener('click', function(e) {
        if (e.target === likeModal) {
            closeLikeModalFunc();
        }
    });

    // 绑定收藏模态框
    if (collectLink) {
        collectLink.addEventListener('click', openCollectModal);
    }
    if (closeCollectModal) {
        closeCollectModal.addEventListener('click', closeCollectModalFunc);
    }
    collectModal.addEventListener('click', function(e) {
        if (e.target === collectModal) {
            closeCollectModalFunc();
        }
    });

    //--------------------------------登陆注册按钮点击----------------------------------------
    // 监听登录注册按钮的点击事件
    if (loginRegisterBtn) {
        loginRegisterBtn.addEventListener('click', function() {
            // 显示登录注册模态框
            loginRegisterModal.classList.remove('hidden');
            document.body.style.overflow = 'hidden'; // 禁止页面滚动
        });
    }
    // 监听模态框关闭按钮的点击事件
    const closeLoginRegisterModal = document.getElementById('close-login-register-modal');
    if (closeLoginRegisterModal) {
        closeLoginRegisterModal.addEventListener('click', function() {
            // 隐藏登录注册模态框
            loginRegisterModal.classList.add('hidden');
            document.body.style.overflow = ''; // 恢复页面滚动
        });
    }
    // 监听模态框外部点击事件，关闭模态框
    loginRegisterModal.addEventListener('click', function(e) {
        if (e.target === loginRegisterModal) {
            loginRegisterModal.classList.add('hidden');
            document.body.style.overflow = ''; // 恢复页面滚动
        }
    });
    // 监听登录选项卡的点击事件
    if (loginTab) {
        loginTab.addEventListener('click', function() {
            // 切换选项卡样式
            loginTab.classList.add('text-primary', 'border-primary');
            loginTab.classList.remove('text-gray-500', 'border-transparent');
            registerTab.classList.add('text-gray-500', 'border-transparent');
            registerTab.classList.remove('text-primary', 'border-primary');

            // 显示登录表单，隐藏注册表单
            loginPanel.classList.remove('hidden');
            registerPanel.classList.add('hidden');
        });
    }
    // 监听注册选项卡的点击事件
    if (registerTab) {
        registerTab.addEventListener('click', function() {
            // 切换选项卡样式
            registerTab.classList.add('text-primary', 'border-primary');
            registerTab.classList.remove('text-gray-500', 'border-transparent');
            loginTab.classList.add('text-gray-500', 'border-transparent');
            loginTab.classList.remove('text-primary', 'border-primary');

            // 显示注册表单，隐藏登录表单
            registerPanel.classList.remove('hidden');
            loginPanel.classList.add('hidden');
        });
    }
    
    checkLoginStatus();

    initCommentFunctionality();

    initPostDeleteFunctionality();

    initCollectFunctionality();

    initFollowFunctionality();

    initImageUpload();

    initAvatarUpload();

    window.addEventListener('load', function() {
        loadPosts(); // 页面加载完成后自动加载帖子
        loadRecommendUsers(); // 加载推荐用户
    });

    window.addEventListener('scroll', function() {
        if(isLoading || !hasMorePosts) return;

        const scrollHeight = document.documentElement.scrollHeight;
        const scrollTop = document.body.scrollTop || document.documentElement.scrollTop;
        const clientHeight = document.documentElement.clientHeight;

        // 当滚动到底部距离200时加载更多帖子
        if (scrollTop + clientHeight >= scrollHeight - 200) {
            currentPage++;
            loadPosts();
        }
    });


    // 更新视图按钮状态
    function updateViewButtons() {
        allPostsBtn.classList.remove('text-gray-500', 'border-transparent');
        allPostsBtn.classList.add('text-primary', 'border-b-2', 'border-primary');
        
        followedPostsBtn.classList.remove('text-primary', 'border-b-2', 'border-primary');
        followedPostsBtn.classList.add('text-gray-500', 'border-transparent');
        
        if (currentView === 'followed') {
            allPostsBtn.classList.remove('text-primary', 'border-b-2', 'border-primary');
            allPostsBtn.classList.add('text-gray-500', 'border-transparent');
            
            followedPostsBtn.classList.remove('text-gray-500', 'border-transparent');
            followedPostsBtn.classList.add('text-primary', 'border-b-2', 'border-primary');
        }
    }

    // 加载帖子列表
    async function loadPosts() {
        if (isLoading) return;

        isLoading = true;
        if(currentPage == 1){
            postList.innerHTML = '';  // 清空帖子列表
            noPosts.classList.add('hidden');  // 隐藏无帖子提示
            loadIndicator.classList.remove('hidden');  // 显示加载指示器
        }  
        else{
            //显示加载指示器
            const loadingE1 = document.createElement('div');
            loadingE1.className = 'text-center py-8';
            loadingE1.id = "temp-loading";
            loadingE1.innerHTML = `
                <i class="fa fa-spinner fa-spin text-primary"></i>
                <p class="mt-2 text-gray-500">加载更多动态...</p>
            `;
            postList.appendChild(loadingE1);
        }
        try {
            let apiUrl = `http://localhost:3000/api/posts?page=${currentPage}&userId=${globalUser.id}`;
            if (currentView === 'followed') {
                // 关注视图调用专门的API
                apiUrl = `http://localhost:3000/api/followed_posts?page=${currentPage}&userId=${globalUser.id}`;
            }
            const response = await fetch(apiUrl);
            if (!response.ok) throw new Error('API请求失败');
            
            const result = await response.json();
            renderPosts(result.data);
            
            // 更新分页状态
            hasMorePosts = result.pagination.hasMore;
            
            // 移除临时加载指示器
            if (currentPage > 1) {
                document.getElementById('temp-loading')?.remove();
            }
            
            // 显示空状态
            if (currentPage === 1 && result.data.length === 0) {
                noPosts.classList.remove('hidden');
            }
        } catch (error) {
            console.error('加载帖子失败:', error);
            if (currentPage === 1) {
                noPosts.classList.remove('hidden');
                noPosts.textContent = '加载动态失败，请稍后重试';
            } else {
                alert('加载更多失败，请稍后重试');
            }
        } finally {
            isLoading = false;
            loadIndicator.classList.add('hidden');
        }
    }

    // 渲染帖子列表
    function renderPosts(posts) {
        posts.forEach(post => {
            const postItem = document.createElement('div');
            postItem.className = 'post-item bg-white rounded-xl shadow-sm overflow-hidden animate-fadeIn';
            postItem.style.animationDelay='0.2s';  
            postItem.dataset.postId = post.post_id; //设置帖子的ID
            postItem.dataset.userId = post.user_id; //设置用户ID
             // 处理媒体图片（当media_url存在且不为空数组时显示）
            const mediaHtml = post.media_url && post.media_url.length > 0 
            ? `<div class="overflow-x-auto pb-3">
                <div class="flex space-x-2 min-w-max">
                    ${post.media_url.map(url => `
                        <div class="w-32 h-32 shrink-0 relative cursor-pointer" onclick="openImagePreview('${url}')">
                        <img src="${url}" alt="帖子图片" class="w-full h-full object-cover rounded">
                        <div class="absolute inset-0 bg-black/30 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
                            <i class="fa fa-search-plus text-white text-2xl"></i>
                        </div>
                        </div>
                    `).join('')}
                </div>
            </div>`
            : '';
            postItem.innerHTML = `
                <div class="p-5 relative">
                    <div class="flex items-center space-x-3 mb-3">
                        <a href="Personal.html?userId=${post.user_id}&LoginUserId=${globalUser.id}&LoginUserName=${globalUser.username}&LoginAvatarURL=${globalUser.avatar_url}" class="flex items-center space-x-3">
                            <img src="${post.avatar_url}" alt="用户头像" class="w-10 h-10 rounded-full object-cover">
                        </a>
                        <div>
                            <h4 class="text-lg font-medium">${post.username}</h4>
                            <p class="text-xs text-gray-500">${post.school_info}</p>
                            <span class="text-xs text-gray-500">${formatTimeAgo(post.updated_at)}</span>
                        </div>
                        <button class="ml-auto text-gray-400 hover:text-gray-600 post-actions-btn" data-post-id="${post.post_id}">
                            <i class="fa fa-ellipsis-h"></i>
                        </button>
                    </div>
                    <p class="text-gray-700 mb-3">${post.content}</p>
                    ${mediaHtml}
                    <div class="flex items-center justify-between mt-4">
                        <div class="flex items-center space-x-2">
                            <button class="text-${post.is_liked ? 'primary' : 'gray-500'} hover:text-primary-dark" data-post-id="${post.post_id}" data-is-liked="${post.is_liked}">
                                <i class="fa ${post.is_liked ? 'fa-heart' : 'fa-heart-o'}"></i>
                                <span class="ml-1">${post.like_count}</span>
                            </button>
                            <button class="text-gray-500 hover:text-gray-700 comment-toggle" data-post-id="${post.post_id}">
                                <i class="fa fa-comment-o"></i>
                                <span class="ml-1">${post.comment_count}</span>
                            </button>
                        </div>
                    </div>  

                    <!-- 评论区域（默认隐藏） -->
                    <div class="mt-4 pt-4 border-t border-gray-100 hidden comments-container" data-post-id="${post.post_id}">
                        <div class="flex space-x-2 mb-3 comments-input-container">
                            <img src="${globalUser.avatar_url}" alt="用户头像" class="w-8 h-8 rounded-full object-cover">
                            <div class="flex-1 relative">
                                <input type="text" class="w-full px-3 py-2 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-primary/50 comment-input" placeholder="添加评论...">
                                <button class="absolute right-3 top-1/2 -translate-y-1/2 text-primary">
                                    <i class="fa fa-paper-plane"></i>
                                </button>
                            </div>
                        </div>
                        
                        <h4 class="text-sm font-medium text-gray-700 mb-2">评论 (${post.comment_count})</h4>
                        <div class="comments-list" data-post-id="${post.post_id}">
                            <!-- 评论列表将通过JS动态添加 -->
                        </div>
                        <button class="w-full text-sm text-primary hover:text-primary-dark mt-2 load-more-comments" data-post-id="${post.post_id}">
                            加载更多评论
                        </button>
                    </div>

                    <!-- 操作菜单（默认隐藏） -->
                    <div class="post-actions-menu absolute top-3 right-3 bg-white rounded-lg shadow-lg z-10 hidden transition-all duration-200" data-post-id="${post.post_id}">
                        <ul class="py-1">
                            <li class="px-4 py-2 hover:bg-gray-100 cursor-pointer flex items-center">
                                <i class="fa ${post.is_collected ? 'fa-bookmark' : 'fa-bookmark-o'} mr-2 text-gray-600 w-5 text-center"></i>
                                <span>${post.is_collected ? '已收藏' : '收藏'}</span>
                            </li>
                            <li class="px-4 py-2 hover:bg-gray-100 cursor-pointer flex items-center">
                                <i class="fa ${post.is_following ? 'fa-user' : 'fa-user-plus'} mr-2 text-gray-600 w-5 text-center"></i>
                                <span>${post.is_following ? '已关注' : '关注'}</span>
                            </li>
                            ${post.user_id === globalUser.id || globalUser.type === 1 ? `
                            <li class="px-4 py-2 hover:bg-gray-100 cursor-pointer flex items-center text-danger">
                                <i class="fa fa-trash-o mr-2 w-5 text-center"></i>
                                <span>删除</span>
                            </li>
                            ` : ''}
                        </ul>
                    </div>
                </div>`;
            postList.appendChild(postItem);
        });
        // 如果没有帖子，显示无帖子提示
        if (posts.length === 0 && currentPage === 1) {
            noPosts.classList.remove('hidden');
        } else {
            noPosts.classList.add('hidden');
        }
        // 如果没有更多帖子，隐藏加载更多按钮
        if (!hasMorePosts) {
            loadMoreBtn.classList.add('hidden');
        } else {
            loadMoreBtn.classList.remove('hidden');
        }
    }

    // 发布动态模态框控制
    document.getElementById('post-modal-btn').addEventListener('click', function() {
        if (checkLoginStatusAndPrompt()) return; 
        postModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    });

    document.getElementById('main-post-box').addEventListener('click', function() {
        if (checkLoginStatusAndPrompt()) return; 
        postModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    });

    document.getElementById('close-post-modal').addEventListener('click', function() {
        postModal.classList.add('hidden');
        document.body.style.overflow = '';
    });

    document.getElementById('post-cancel').addEventListener('click', function() {
        postModal.classList.add('hidden');
        document.body.style.overflow = '';
    });

    // 点击模态框外部关闭
    postModal.addEventListener('click', function(e) {
        if (e.target === postModal) {
        postModal.classList.add('hidden');
        document.body.style.overflow = '';
        }
    });
    // 发布动态表单提交
    postForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        try {
            const imageFiles = [];
            const uploadBoxes = imageUploadContainer.querySelectorAll('.image-upload');
            
            uploadBoxes.forEach(box => {
                // 查找当前上传框中的文件输入框（包括预览后的结构）
                const fileInput = box.querySelector('input.image-input');
                if (fileInput && fileInput.files && fileInput.files.length > 0) {
                    imageFiles.push(fileInput.files[0]);
                } else {
                    // 检查预览状态下是否有隐藏的文件引用
                    const fileData = box._file;
                    if (fileData) {
                        imageFiles.push(fileData);
                    }
                }
            });
            console.log('选择的图片文件:', imageFiles);
            // 上传图片并获取路径
            const mediaUrls = await uploadImages(imageFiles);
            console.log('上传的图片路径:', mediaUrls);
            // 发布动态
            await publishPost(mediaUrls);
        } catch (error) {
            console.error('发布动态失败:', error);
            alert(error.message || '发布失败，请稍后重试');
        }
    });

    // 上传图片并返回路径数组
    async function uploadImages(files) {
        if (files.length === 0) return [];

        const formData = new FormData();
        files.forEach(file => {
            // 统一使用"images"字段名，后端通过数组接收
            formData.append('images', file);
        });

        // 显示上传中状态
        const uploadIndicator = document.createElement('div');
        uploadIndicator.className = 'fixed inset-0 z-50 flex items-center justify-center bg-black/50';
        uploadIndicator.innerHTML = `
            <div class="bg-white p-6 rounded-lg shadow-xl">
                <div class="flex items-center">
                    <i class="fa fa-spinner fa-spin text-primary mr-3"></i>
                    <span>正在上传图片...</span>
                </div>
            </div>
        `;
        document.body.appendChild(uploadIndicator);
        try {
            const response = await fetch('http://localhost:3000/api/upload_images', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('图片上传失败');
            }
            
            const result = await response.json();
            console.log('图片上传成功，路径:', result.urls);
            return result.urls || [];
        } catch (error) {
            console.error('图片上传失败:', error);
            throw error;
        } finally {
            // 移除上传中状态
            document.body.removeChild(uploadIndicator);
        }
    }

    // 发布动态
    async function publishPost(mediaUrls) {
        const content = postForm.querySelector('textarea').value.trim();
        if (!content) {
            throw new Error('请输入动态内容');
        }
        
        try {
            // 创建表单数据
            const formData = new FormData();
            formData.append('user_id', globalUser.id); // 实际应用中应从登录状态获取
            formData.append('content', content);
            formData.append('location_info', ''); // 实际应用中应获取位置
            
            // 添加图片路径
            if (mediaUrls && mediaUrls.length > 0) {
                formData.append('media_url', JSON.stringify(mediaUrls));
            }
            
            // 调用API
            const response = await fetch('http://localhost:3000/api/create_posts', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                alert('动态发布成功');
                postModal.classList.add('hidden');
                document.body.style.overflow = '';
                
                // 重置表单
                postForm.reset();
                resetImageUpload();
                
                // 刷新动态列表
                currentPage = 1;
                loadPosts();
            } else {
                const error = await response.json();
                throw new Error(error.message || '发布失败，请稍后重试');
            }
        } catch (error) {
            console.error('发布动态失败:', error);
            throw error;
        }
    }

    // 初始化图片上传功能
    function initImageUpload() {
        // 监听图片上传区域点击事件
        imageUploadContainer.addEventListener('click', function(e) {
            const uploadBox = e.target.closest('.image-upload');
            if (!uploadBox) return;
            
            const fileInput = uploadBox.querySelector('.image-input');
            if (fileInput) fileInput.click();
        });
        
        // 监听文件选择事件
        imageUploadContainer.addEventListener('change', function(e) {
            if (e.target.type !== 'file') return;
            
            const fileInput = e.target;
            const uploadBox = fileInput.closest('.image-upload');
            const file = fileInput.files[0];
            
            if (file) {
                handleImageUpload(file, uploadBox);
            }
        });
    }

    // 处理图片上传
    function handleImageUpload(file, uploadBox) {
        // 检查文件类型
        if (!file.type.startsWith('image/')) {
            alert('请选择图片文件');
            return;
        }
        
        // 检查文件大小（示例：限制为5MB）
        const maxSize = 5 * 1024 * 1024; // 5MB
        if (file.size > maxSize) {
            alert('图片大小不能超过5MB');
            return;
        }

        // 创建图片预览
        const reader = new FileReader();
        reader.onload = function(e) {
            uploadBox._file = file; // 保存文件引用以便后续使用
            // 替换上传框为图片预览
            uploadBox.innerHTML = `
                <div class="relative">
                    <img src="${e.target.result}" alt="预览图" class="w-full h-full object-cover rounded-lg">
                    <button type="button" class="absolute top-1 right-1 w-6 h-6 bg-white/80 rounded-full flex items-center justify-center text-gray-600 hover:text-danger transition-colors remove-image">
                        <i class="fa fa-times"></i>
                    </button>
                    <!-- 隐藏的文件输入框，保留文件引用 -->
                    <input type="file" class="hidden image-input" accept="image/*" value="${file.name}">
                </div>
            `;
            // 显示下一个上传框
            const nextUploadBox = uploadBox.nextElementSibling;
            if (nextUploadBox && nextUploadBox.classList.contains('hidden')) {
                nextUploadBox.classList.remove('hidden');
            }
        
            // 绑定删除图片事件
            uploadBox.querySelector('.remove-image').addEventListener('click', function() {
                removeImage(uploadBox);
            });
        }
        reader.readAsDataURL(file);
    }
    // 移除图片
    function removeImage(uploadBox) {
        // 重置上传框
        uploadBox.innerHTML = `
            <i class="fa fa-image text-gray-400 text-xl"></i>
            <p class="text-xs text-gray-500 mt-1">添加图片</p>
            <input type="file" class="hidden image-input" accept="image/*">
        `;
        
        // 隐藏多余的上传框，保持最多4个可见
        const allUploadBoxes = imageUploadContainer.querySelectorAll('.image-upload');
        let visibleCount = 0;
        
        allUploadBoxes.forEach(box => {
            if (!box.classList.contains('hidden')) {
                visibleCount++;
            }
        });
        
        // 如果有多余的空上传框，隐藏它们
        if (visibleCount > 1) {
            let foundContent = false;
            allUploadBoxes.forEach(box => {
                if (!box.classList.contains('hidden')) {
                    if (foundContent && box.querySelector('input[type="file"]')) {
                        box.classList.add('hidden');
                    }
                    foundContent = true;
                }
            });
        }
    }
    // 重置图片上传框
    function resetImageUpload() {
        const uploadBoxes = imageUploadContainer.querySelectorAll('.image-upload');
        
        uploadBoxes.forEach((box, index) => {
            if (index === 0) {
                // 第一个上传框保持可见
                box.innerHTML = `
                    <i class="fa fa-image text-gray-400 text-xl"></i>
                    <p class="text-xs text-gray-500 mt-1">添加图片</p>
                    <input type="file" class="hidden image-input" accept="image/*">
                `;
                box.classList.remove('hidden');
            } else {
                // 其他上传框隐藏
                box.innerHTML = `
                    <i class="fa fa-image text-gray-400 text-xl"></i>
                    <p class="text-xs text-gray-500 mt-1">添加图片</p>
                    <input type="file" class="hidden image-input" accept="image/*">
                `;
                box.classList.add('hidden');
            }
        });
    }

    // 帖子的点赞和取消点赞
    document.addEventListener('click', async function(e) {
        // 处理点赞
        const likeBtn = e.target.closest('.fa-heart-o')?.parentElement;
        if (likeBtn) {
            if (checkLoginStatusAndPrompt()) return; 
            const postItem = likeBtn.closest('.post-item');
            const postId = postItem.dataset.postId;
            const countSpan = likeBtn.querySelector('span');
            const count = parseInt(countSpan.textContent);
            
            likeBtn.classList.add('scale-110');
            setTimeout(() => likeBtn.classList.remove('scale-110'), 200);

            try {
                const response = await fetch(`http://localhost:3000/api/like/${postId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ user_id: globalUser.id })
                });
                
                if (response.ok) {
                    const result = await response.json();
                    // 更新UI
                    likeBtn.querySelector('i').classList.remove('fa-heart-o');
                    likeBtn.querySelector('i').classList.add('fa-heart');
                    likeBtn.classList.add('text-primary');
                    countSpan.textContent = result.likeCount;
                } else {
                    throw new Error('点赞失败');
                }
            } catch (error) {
                console.error('点赞失败:', error);
                alert('点赞失败，请稍后重试');
            }
        }
        else{
            // 处理取消点赞
            const unlikeBtn = e.target.closest('.fa-heart')?.parentElement;
            if (unlikeBtn) {
                const postItem = unlikeBtn.closest('.post-item');
                const postId = postItem.dataset.postId;
                const countSpan = unlikeBtn.querySelector('span');
                const count = parseInt(countSpan.textContent);
                
                // 添加点击动画
                unlikeBtn.classList.add('scale-110');
                setTimeout(() => unlikeBtn.classList.remove('scale-110'), 200);

                try {
                    const response = await fetch(`http://localhost:3000/api/like/${postId}`, {
                        method: 'DELETE',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ user_id: globalUser.id })
                    });
                    console.log(response)
                    
                    if (response.ok) {
                        const result = await response.json();
                        // 更新UI
                        unlikeBtn.querySelector('i').classList.remove('fa-heart');
                        unlikeBtn.querySelector('i').classList.add('fa-heart-o');
                        unlikeBtn.classList.remove('text-primary');
                        countSpan.textContent = result.likeCount;
                    } else {
                        throw new Error('取消点赞失败');
                    }
                } catch (error) {
                    console.error('取消点赞失败:', error);
                    alert('取消点赞失败，请稍后重试');
                }
            }
         }
    });         
    
    // 初始化评论区
    function initCommentFunctionality() {
        // 监听评论区切换
        document.addEventListener('click', function(e) {
            const commentToggle = e.target.closest('.comment-toggle');
            if (commentToggle) {
                e.preventDefault();
                const postId = commentToggle.dataset.postId;
                const commentsContainer = document.querySelector(`.comments-container[data-post-id="${postId}"]`);
                const inputContainer = commentsContainer.querySelector('.comments-input-container');
                // 滚动到评论区域
                const input = inputContainer.querySelector('.comment-input');
                
                if (commentsContainer.classList.contains('hidden')) {
                    // 显示评论区
                    commentsContainer.classList.remove('hidden');
                    // 聚焦输入框并滚动
                    if (input) {
                        input.focus();
                        inputContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    }
                    // 加载评论
                    loadComments(postId, 1);
                } else {
                    // 隐藏评论区
                    commentsContainer.classList.add('hidden');
                    // 重置输入框
                    if (input) {
                        input.value = '';
                        input.blur();
                    }
                }
            }
        });
        
        // 提交评论
        document.addEventListener('click', function(e) {
            const submitBtn = e.target.closest('.comment-submit');
            if (submitBtn) {
                if (checkLoginStatusAndPrompt()) return; 
                e.preventDefault();
                handleCommentSubmit(submitBtn);
            }
        });
        
        // 键盘提交评论（Enter键）
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && e.target.closest('.comment-input')) {
                e.preventDefault();
                if (checkLoginStatusAndPrompt()) return; 
                handleCommentSubmit(e.target);
            }
        });
        
        // 加载更多评论
        document.addEventListener('click', function(e) {
            const loadMoreBtn = e.target.closest('.load-more-comments');
            if (loadMoreBtn) {
                e.preventDefault();
                const postId = loadMoreBtn.dataset.postId;
                const currentPage = parseInt(loadMoreBtn.dataset.page) || 1;
                loadComments(postId, currentPage + 1, loadMoreBtn);
            }
        });

        // 点赞评论
        document.addEventListener('click', async function(e) {
            // 处理点赞（未点赞状态）
            const likeBtn = e.target.closest('.comments-container .fa-thumbs-o-up')?.parentElement;
            if (likeBtn) {
                if (checkLoginStatusAndPrompt()) return; 
                const commentId = likeBtn.dataset.commentId;
                const countSpan = likeBtn.querySelector('span');
                const currentCount = parseInt(countSpan.textContent);
                
                // 防止重复点击
                // if (likeBtn.disabled) return;
                // likeBtn.disabled = true;
                // likeBtn.classList.add('opacity-50', 'cursor-not-allowed');
                
                try {
                    const response = await fetch(`http://localhost:3000/api/comment/like/${commentId}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ user_id: globalUser.id })
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        // 更新UI
                        likeBtn.querySelector('i').classList.remove('fa-thumbs-o-up');
                        likeBtn.querySelector('i').classList.add('fa-thumbs-up');
                        likeBtn.classList.remove('text-gray-500');
                        likeBtn.classList.add('text-primary');
                        countSpan.textContent = result.likeCount;
                    } else {
                        throw new Error('点赞失败');
                    }
                } catch (error) {
                    console.error('点赞失败:', error);
                    alert('点赞失败，请稍后重试');
                    // 回滚UI状态
                    likeBtn.querySelector('i').classList.remove('fa-thumbs-up');
                    likeBtn.querySelector('i').classList.add('fa-thumbs-o-up');
                    likeBtn.classList.remove('text-primary');
                    likeBtn.classList.add('text-gray-500');
                } finally {
                    likeBtn.disabled = false;
                    likeBtn.classList.remove('opacity-50', 'cursor-not-allowed');
                }
            }
            else{
                // 处理取消点赞（已点赞状态）
                const unlikeBtn = e.target.closest('.comments-container .fa-thumbs-up')?.parentElement;
                if (unlikeBtn) {
                    const commentId = unlikeBtn.dataset.commentId;
                    const countSpan = unlikeBtn.querySelector('span');
                    const currentCount = parseInt(countSpan.textContent);
                    
                    // 防止重复点击
                    // if (unlikeBtn.disabled) return;
                    // unlikeBtn.disabled = true;
                    // unlikeBtn.classList.add('opacity-50', 'cursor-not-allowed');
                    
                    try {
                        const response = await fetch(`http://localhost:3000/api/comment/like/${commentId}`, {
                            method: 'DELETE',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ user_id: globalUser.id })
                        });
                        
                        if (response.ok) {
                            const result = await response.json();
                            // 更新UI
                            unlikeBtn.querySelector('i').classList.remove('fa-thumbs-up');
                            unlikeBtn.querySelector('i').classList.add('fa-thumbs-o-up');
                            unlikeBtn.classList.remove('text-primary');
                            unlikeBtn.classList.add('text-gray-500');
                            countSpan.textContent = result.likeCount;
                        } else {
                            throw new Error('取消点赞失败');
                        }
                    } catch (error) {
                        console.error('取消点赞失败:', error);
                        alert('取消点赞失败，请稍后重试');
                        // 回滚UI状态
                        unlikeBtn.querySelector('i').classList.remove('fa-thumbs-o-up');
                        unlikeBtn.querySelector('i').classList.add('fa-thumbs-up');
                        unlikeBtn.classList.remove('text-gray-500');
                        unlikeBtn.classList.add('text-primary');
                    } finally {
                        unlikeBtn.disabled = false;
                        unlikeBtn.classList.remove('opacity-50', 'cursor-not-allowed');
                    }
                }
            }    
        });

        // 删除评论
        document.addEventListener('click', function(e) {
            const deleteBtn = e.target.closest('.comment-delete');
            if (deleteBtn) {
                e.preventDefault();
                const commentId = deleteBtn.dataset.commentId;
                const postId = deleteBtn.dataset.postId;
                const commentItem = deleteBtn.closest('.mb-3');
                
                // 确认删除
                if (confirm('确定要删除这条评论吗？此操作不可恢复。')) {
                    deleteComment(commentId, postId, commentItem);
                }
            }
        });
    }

    // 处理评论提交
    function handleCommentSubmit(target) {
        const inputContainer = target.closest('.comments-input-container');
        const commentsContainer = inputContainer.closest('.comments-container');
        const postId = commentsContainer.dataset.postId;
        const input = inputContainer.querySelector('.comment-input');
        const content = input.value.trim();
        
        if (content) {
            addComment(postId, content);
            input.value = '';
        }
    }

    // 加载评论
    async function loadComments(postId, page, loadMoreBtn = null) {
        try {
            const response = await fetch(`http://localhost:3000/api/comments/${postId}?page=${page}&userId=${globalUser.id}`);
            if (!response.ok) throw new Error('获取评论失败');
            
            const result = await response.json();
            renderComments(postId, result.data, page, result.pagination.hasMore, loadMoreBtn);
        } catch (error) {
            console.error('加载评论失败:', error);
            alert('加载评论失败，请稍后重试');
        }
    }

    // 渲染评论
    function renderComments(postId, comments, page, hasMore, loadMoreBtn) {
        const commentsList = document.querySelector(`.comments-list[data-post-id="${postId}"]`);
        
        if (page === 1) {
            commentsList.innerHTML = '';
        }
        
        if (comments.length === 0) {
            if (page === 1) {
                commentsList.innerHTML = '<p class="text-sm text-gray-500">还没有评论，快来发表第一条评论吧</p>';
            }
            return;
        }
        comments.forEach(comment => {
            const commentElement = document.createElement('div');
            commentElement.className = 'mb-3';

            const showDeleteBtn = comment.user_id === globalUser.id || globalUser.type === 1;
            const deleteBtn = showDeleteBtn ? `
                <button class="text-xs text-danger hover:text-danger-dark ml-2 comment-delete" 
                        data-comment-id="${comment.comment_id}" 
                        data-post-id="${postId}">
                    <i class="fa fa-trash-o"></i> 删除
                </button>
            ` : '';
            commentElement.innerHTML = `
                <div class="flex space-x-2">
                    <img src="${comment.avatar_url}" alt="评论用户头像" class="w-8 h-8 rounded-full object-cover">
                    <div class="flex-1">
                        <div class="flex items-center space-x-1">
                            <h5 class="text-sm font-medium">${comment.username}</h5>
                            <span class="text-xs text-gray-500">${formatTimeAgo(comment.updated_at)}</span>
                        </div>
                        <p class="text-sm text-gray-700 mt-1">${comment.content}</p>
                        <div class="flex items-center mt-1">
                            <button class="text-xs text-${comment.is_liked ? 'primary' : 'gray-500'} hover:text-primary-dark comment-like-btn" data-comment-id="${comment.comment_id}" data-is-liked="${comment.is_liked}">
                                <i class="fa ${comment.is_liked ? 'fa-thumbs-up' : 'fa-thumbs-o-up'}"></i>
                                <span class="ml-1">${comment.like_count}</span>
                            </button>
                            ${deleteBtn}
                        </div>
                    </div>
                </div>
            `;
            commentsList.appendChild(commentElement);
        });
        
        if (loadMoreBtn) {
            loadMoreBtn.dataset.page = page;
            if (hasMore) {
                loadMoreBtn.innerHTML = '加载更多评论';
            } else {
                loadMoreBtn.innerHTML = '没有更多评论了';
                loadMoreBtn.disabled = true;
                loadMoreBtn.classList.add('text-gray-400', 'cursor-not-allowed');
            }
        }
    }

    // 添加评论
    async function addComment(postId, content) {
        try {
            const response = await fetch(`http://localhost:3000/api/create_comments/${postId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: globalUser.id,
                    content: content
                })
            });
            
            if (response.ok) {
                const result = await response.json();
                const commentsContainer = document.querySelector(`.comments-container[data-post-id="${postId}"]`);
                const commentCountBtn = commentsContainer.closest('.post-item').querySelector('.comment-toggle span');
                
                // 更新评论数
                const currentCount = parseInt(commentCountBtn.textContent);
                commentCountBtn.textContent = currentCount + 1;
                
                // 更新评论区内部的标题计数
                const innerCommentCount = commentsContainer.querySelector('h4');
                if (innerCommentCount) {
                    const currentInnerCount = parseInt(innerCommentCount.textContent.match(/\((\d+)\)/)[1]);
                    innerCommentCount.textContent = `评论 (${currentInnerCount + 1})`;
                }

                // 显示成功提示
                alert('评论添加成功');

                // 重新加载评论列表（关键修改点）
                const loadMoreBtn = commentsContainer.querySelector('.load-more-comments');
                loadComments(postId, 1, loadMoreBtn);
            } else {
                const error = await response.json();
                throw new Error(error.message || '添加评论失败');
            }
        } catch (error) {
            console.error('添加评论失败:', error);
            alert(error.message || '添加评论失败，请稍后重试');
        }
    }
    
    // 帖子菜单功能操作
    document.addEventListener('click', function(e) {
        const actionsBtn = e.target.closest('.post-actions-btn');
        if (!actionsBtn) return;
        if (checkLoginStatusAndPrompt()) return; 
        const postId = actionsBtn.dataset.postId;
        const postItem = actionsBtn.closest('.post-item');
        
        // 隐藏其他帖子的菜单
        document.querySelectorAll('.post-actions-menu').forEach(menu => {
            if (menu.dataset.postId !== postId) {
                menu.classList.add('hidden');
            }
        });
        
        // 显示当前帖子的菜单
        const menu = postItem.querySelector('.post-actions-menu');
        menu.classList.toggle('hidden');
        
        if (menu.classList.contains('hidden')) return;
        
        // 确保菜单相对于帖子内容容器定位
        const contentContainer = postItem.querySelector('.p-5');
        if (contentContainer) {
            // 重置可能的样式
            menu.style.top = '';
            menu.style.right = '';
            
            // 使用Tailwind类设置位置
            menu.classList.add('top-3', 'right-3');
        }
    });

    // 点击其他地方关闭菜单
    document.addEventListener('click', function(e) {
        const actionsBtn = e.target.closest('.post-actions-btn');
        const menu = e.target.closest('.post-actions-menu');
        
        if (!actionsBtn && !menu) {
            document.querySelectorAll('.post-actions-menu').forEach(menu => {
                menu.classList.add('hidden');
            });
        }
    });

    // 初始化帖子的删除功能
    function initPostDeleteFunctionality() {
        document.addEventListener('click', function(e) {
            if(!e.target.closest('.post-actions-menu .fa-trash-o')) return;
            const deleteOption = e.target.closest('.post-actions-menu .fa-trash-o').parentElement;
            if (!deleteOption) return;
            
            // 获取帖子ID
            const postActionsMenu = deleteOption.closest('.post-actions-menu');
            const postId = postActionsMenu.dataset.postId;
            const postItem = postActionsMenu.closest('.post-item');
            
            // 显示删除确认对话框
            if (confirm('确定要删除这条动态吗？此操作不可恢复。')) {
                deletePost(postId, postItem);
            }
        });
    }

    // 删除帖子
    async function deletePost(postId, postItem) {
        try {
            console.log(postItem.getAttribute('data-user-id'));
            // 调用删除API
            const response = await fetch(`http://localhost:3000/api/delete_posts/${postId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                },
                // 从帖子中获取用户ID
                body: JSON.stringify({ userId: postItem.getAttribute('data-user-id') })
            });
            
            if (!response.ok) {
                throw new Error('删除失败');
            }
            
            // 删除成功，从UI中移除帖子
            if (postItem && postItem.parentNode) {
                postItem.remove();
            }
        
            // 显示成功提示
            alert('动态删除成功');

            if(isLoading) return; // 如果正在加载，则不执行后续操作
            currentPage = 1; // 重置到第一页
            loadPosts(); // 重新加载所有帖子
        } catch (error) {
            console.error('删除动态失败:', error);
            alert('删除失败，请稍后重试');
        } 
    }

    // 删除评论
    async function deleteComment(commentId, postId, commentItem) {
        try {
            // 调用删除评论API
            const response = await fetch(`http://localhost:3000/api/delete_comments/${commentId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error('删除失败');
            }
            
            // 删除成功，从UI中移除评论
            if (commentItem && commentItem.parentNode) {
                commentItem.remove();
            }
            
            // 更新评论数
            const commentsContainer = document.querySelector(`.comments-container[data-post-id="${postId}"]`);
            const commentCountBtn = commentsContainer.closest('.post-item').querySelector('.comment-toggle span');
            const currentCount = parseInt(commentCountBtn.textContent);
            
            if (currentCount > 0) {
                commentCountBtn.textContent = currentCount - 1;
            }
            
            // 更新评论区内部的标题计数
            const innerCommentCount = commentsContainer.querySelector('h4');
            if (innerCommentCount) {
                const currentInnerCount = parseInt(innerCommentCount.textContent.match(/\((\d+)\)/)[1]);
                innerCommentCount.textContent = `评论 (${Math.max(0, currentInnerCount - 1)})`;
            }
            // 显示成功提示
            alert('评论删除成功');
            
            // 检查是否还有评论，没有则显示提示
            const commentsList = commentsContainer.querySelector('.comments-list');
            if (commentsList.children.length === 0) {
                commentsList.innerHTML = '<p class="text-sm text-gray-500">还没有评论，快来发表第一条评论吧</p>';
            }
        } catch (error) {
            console.error('删除评论失败:', error);
            alert('删除失败，请稍后重试');
        } 
    }

    // 初始化收藏功能
    function initCollectFunctionality() {
        // 帖子操作菜单中的收藏选项功能
        document.addEventListener('click', function(e) {
            // 选择操作菜单中的收藏选项（第一个li元素）
            const collectOption = e.target.closest('.post-actions-menu li:first-child');
            if (collectOption) {
                e.preventDefault();
                const postActionsMenu = collectOption.closest('.post-actions-menu');
                const postId = postActionsMenu.dataset.postId;
                const postItem = postActionsMenu.closest('.post-item');
                const isCollected = collectOption.querySelector('i').classList.contains('fa-bookmark');
                
                // 调用收藏/取消收藏API
                if (isCollected) {
                    // 取消收藏
                    fetch(`http://localhost:3000/api/collect/${postId}`, {
                        method: 'DELETE',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ user_id: globalUser.id })
                    })
                    .then(response => {
                        if (!response.ok) throw new Error('取消收藏失败');
                        return response.json();
                    })
                    .then(result => {
                        // 更新菜单文本和图标
                        collectOption.querySelector('i').classList.remove('fa-bookmark','text-green-500');
                        collectOption.querySelector('i').classList.add('fa-bookmark-o','text-gray-600');
                        collectOption.querySelector('span').textContent = '收藏';
                        alert('已取消收藏');
                    })
                    .catch(error => {
                        console.error('取消收藏失败:', error);
                        alert('取消收藏失败，请稍后重试');
                    });
                } else {
                    // 收藏
                    fetch(`http://localhost:3000/api/collect/${postId}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ user_id: globalUser.id })
                    })
                    .then(response => {
                        if (!response.ok) throw new Error('收藏失败');
                        return response.json();
                    })
                    .then(result => {
                        // 更新菜单文本和图标
                        collectOption.querySelector('i').classList.remove('fa-bookmark-o','text-gray-600');
                        collectOption.querySelector('i').classList.add('fa-bookmark','text-green-500');
                        collectOption.querySelector('span').textContent = '已收藏';
                        
                        alert('收藏成功');
                    })
                    .catch(error => {
                        console.error('收藏失败:', error);
                        alert('收藏失败，请稍后重试');
                    });
                }
                
                // 关闭操作菜单
                postActionsMenu.classList.add('hidden');
            }
        });
    }

    // 初始化关注功能
    function initFollowFunctionality() {
        // 帖子操作菜单中的关注选项功能
        document.addEventListener('click', function(e) {
            // 选择操作菜单中的关注选项（第二个li元素）
            const followOption = e.target.closest('.post-actions-menu li:nth-child(2)');
            if (followOption) {
                e.preventDefault();
                const postActionsMenu = followOption.closest('.post-actions-menu');
                const postId = postActionsMenu.dataset.postId;
                const postItem = postActionsMenu.closest('.post-item');
                const authorId = postItem.dataset.userId; // 帖子作者ID
                const isFollowing = followOption.querySelector('i').classList.contains('fa-user');
                
                // 添加点击动画
                followOption.classList.add('scale-110');
                setTimeout(() => followOption.classList.remove('scale-110'), 200);
                
                // 调用关注/取消关注API
                if (isFollowing) {
                    // 取消关注
                    fetch(`http://localhost:3000/api/follow/${authorId}`, {
                        method: 'DELETE',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ followerId: globalUser.id })
                    })
                    .then(response => {
                        if (!response.ok) throw new Error('取消关注失败');
                        return response.json();
                    })
                    .then(() => {
                        // 更新UI
                        followOption.querySelector('i').classList.remove('fa-user', 'text-green-500');
                        followOption.querySelector('i').classList.add('fa-user-plus', 'text-gray-600');
                        followOption.querySelector('span').textContent = '关注';
                        alert('已取消关注');
                        
                        // 取消关注后重新加载动态列表
                        if (currentView === 'followed') {
                            currentPage = 1;
                            loadPosts();
                        }
                    })
                    .catch(error => {
                        console.error('取消关注失败:', error);
                        alert('取消关注失败，请稍后重试');
                    });
                } else {
                    // 关注
                    fetch(`http://localhost:3000/api/follow/${authorId}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ followerId: globalUser.id })
                    })
                    .then(response => {
                        if (!response.ok) throw new Error('关注失败');
                        return response.json();
                    })
                    .then(() => {
                        // 更新UI
                        followOption.querySelector('i').classList.remove('fa-user-plus', 'text-gray-600');
                        followOption.querySelector('i').classList.add('fa-user', 'text-green-500');
                        followOption.querySelector('span').textContent = '已关注';
                        alert('关注成功');
                    })
                    .catch(error => {
                        console.error('关注失败:', error);
                        alert('关注失败，请稍后重试');
                    });
                }
                
                // 关闭操作菜单
                postActionsMenu.classList.add('hidden');
            }
        });
    }

    //------------------------------------注册模态框----------------------------------
    // 初始化头像上传功能
    function initAvatarUpload() {
        const avatarUpload = document.getElementById('avatarUpload');
        const avatarPreview = document.getElementById('avatarPreview');
        const avatarImg = document.getElementById('avatarImg');
        const avatarIcon = avatarPreview.querySelector('i.fa-user');
        const removeAvatarBtn = document.getElementById('removeAvatar');
        
        if (!avatarUpload || !avatarPreview || !avatarImg || !avatarIcon || !removeAvatarBtn){
            console.error('头像上传元素未找到，请检查HTML结构');
            return;
        }
        
        // 监听文件选择
        avatarUpload.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (!file) return;
            
            // 检查文件类型
            if (!file.type.startsWith('image/')) {
                alert('请选择图片文件');
                return;
            }
            
            // 检查文件大小（限制为10MB）
            const maxSize = 10 * 1024 * 1024;
            if (file.size > maxSize) {
                alert('图片大小不能超过10MB');
                return;
            }
            
            // 保存文件对象供上传使用
            avatarPreview._avatarFile = file;
            
            // 显示图片预览
            const reader = new FileReader();
            reader.onload = function(e) {
                avatarImg.src = e.target.result;
                avatarImg.classList.remove('hidden');
                avatarIcon.classList.add('hidden');
                removeAvatarBtn.classList.remove('hidden');
                avatarPath.value = ''; // 重置路径，因为还未上传
            };
            reader.onerror = function() {
                alert('读取图片文件时发生错误，请重试。');
            };
            reader.readAsDataURL(file);
        });
        
        // 监听移除按钮
        removeAvatarBtn.addEventListener('click', function() {
            resetAvatar();
        });
    }
    // 重置头像选择
    function resetAvatar() {
        const avatarUpload = document.getElementById('avatarUpload');
        const avatarPreview = document.getElementById('avatarPreview');
        const avatarImg = document.getElementById('avatarImg');
        const avatarIcon = avatarPreview.querySelector('i.fa-user');
        const removeAvatarBtn = document.getElementById('removeAvatar');
        const avatarPath = document.getElementById('avatarPath');
        
        // 重置文件输入
        avatarUpload.value = '';
        delete avatarPreview._avatarFile;
        
        // 重置预览
        avatarImg.src = '';
        avatarImg.classList.add('hidden');
        avatarIcon.classList.remove('hidden');
        removeAvatarBtn.classList.add('hidden');
        avatarPath.value = '';
    }
    // 上传头像并返回路径
    async function uploadAvatar() {
        const avatarPreview = document.getElementById('avatarPreview');
        const avatarPath = document.getElementById('avatarPath');
        
        // 如果没有选择头像，返回null
        if (!avatarPreview._avatarFile) {
            return null;
        }
        
        // 创建表单数据（注意参数名images与现有接口保持一致）
        const formData = new FormData();
        formData.append('images', avatarPreview._avatarFile);
        
        // 显示上传指示器
        const uploadIndicator = document.createElement('div');
        uploadIndicator.className = 'fixed inset-0 z-50 flex items-center justify-center bg-black/50';
        uploadIndicator.innerHTML = `
            <div class="bg-white p-6 rounded-lg shadow-xl">
                <div class="flex items-center">
                    <i class="fa fa-spinner fa-spin text-primary mr-3"></i>
                    <span>正在上传头像...</span>
                </div>
            </div>
        `;
        document.body.appendChild(uploadIndicator);
        
        try {
            // 发送请求到现有的图片上传接口
            const response = await fetch('http://localhost:3000/api/upload_images', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('头像上传失败');
            }
            
            const result = await response.json();
            
            // 确保接口返回了urls数组
            if (!result.urls || result.urls.length === 0) {
                throw new Error('未获取到头像URL');
            }
            
            // 获取第一个上传的图片URL作为头像URL
            const avatarUrl = result.urls[0];
            
            // 保存头像路径
            avatarPath.value = avatarUrl;
            
            return avatarUrl;
        } catch (error) {
            console.error('上传头像失败:', error);
            alert('上传头像失败: ' + error.message);
            return null;
        } finally {
            // 移除上传指示器
            document.body.removeChild(uploadIndicator);
        }
    }
    // 绑定注册表单提交事件
    document.getElementById('registerForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        try {
            // 上传头像（如果有选择）
            const avatarUrl = await uploadAvatar();
            
            // 获取表单数据
            const formData = {
                name: document.getElementById('registerName').value.trim(),
                email: document.getElementById('registerEmail').value.trim(),
                bio: document.getElementById('registerBio').value.trim(),
                password: document.getElementById('registerPassword').value,
                confirmPassword: document.getElementById('registerConfirmPassword').value,
                avatarUrl: avatarUrl || '' // 使用上传的头像路径或空字符串
            };
            
            // 验证表单
            if (!formData.name) {
                throw new Error('请输入昵称');
            }
            
            if (!formData.email) {
                throw new Error('请输入邮箱');
            }
            
            if (!formData.password) {
                throw new Error('请输入密码');
            }
            
            if (formData.password !== formData.confirmPassword) {
                throw new Error('两次输入的密码不一致');
            }
            
            // 发送注册请求
            const response = await fetch('http://localhost:3000/api/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || '注册失败');
            }
            
            const result = await response.json();
            alert('注册成功！请登录');
            
            // 关闭模态框
            document.getElementById('close-login-register-modal').click();
            
            // 重置表单
            document.getElementById('registerForm').reset();
            resetAvatar();
        } catch (error) {
            console.error('注册失败:', error);
            alert(error.message);
        }
    });
    console.log(globalUser.id,globalUser.username, globalUser.avatar_url);
    //-----------------------------个人主页跳转-------------------------------
    const personalHomeBtn = document.getElementById('menu-personal-home');
    if (personalHomeBtn) {
        personalHomeBtn.addEventListener('click', function(e) {
            e.preventDefault();
            if (globalUser.id) {
                window.location.href = `Personal.html?userId=${globalUser.id}&LoginUserId=${globalUser.id}&LoginUserName=${globalUser.username}&LoginAvatarURL=${globalUser.avatar_url}`;
            } else {
                // 未登录状态提示
                alert('请先登录查看个人主页');
                // 显示登录模态框
                loginRegisterModal.classList.remove('hidden');
                document.body.style.overflow = 'hidden';
            }
        });
    }

    // 侧边栏个人主页按钮跳转
    const sidebarPersonalHomeBtn = document.getElementById('sidebar-personal-home');
    if (sidebarPersonalHomeBtn) {
        sidebarPersonalHomeBtn.addEventListener('click', function() {
            if (globalUser.id) {
                window.location.href = `Personal.html?userId=${globalUser.id}&LoginUserId=${globalUser.id}&LoginUserName=${globalUser.username}&LoginAvatarURL=${globalUser.avatar_url}`;
            } else {
                // 未登录状态提示
                alert('请先登录查看个人主页');
                // 显示登录模态框
                loginRegisterModal.classList.remove('hidden');
                document.body.style.overflow = 'hidden';
            }
        });
    }

    //----------------------------关注模态框----------------------------------
    // 打开关注列表模态框
    function openFollowModal() {
        if (checkLoginStatusAndPrompt()) return;
        followLoading.classList.remove('hidden');
        followList.innerHTML = '';
        noFollow.classList.add('hidden');
        followModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
        
        // 加载关注列表
        loadFollowList();
    }
    // 关闭关注列表模态框
    function closeFollowModalFunc() {
        followModal.classList.add('hidden');
        document.body.style.overflow = '';
    }
    // 加载关注列表
    async function loadFollowList() {
        try {
            const response = await fetch(`http://localhost:3000/api/getFollows/${globalUser.id}`);
            if (!response.ok) throw new Error('获取关注列表失败');
            
            const result = await response.json();
            renderFollowList(result.data);
        } catch (error) {
            console.error('加载关注列表失败:', error);
            alert('加载关注列表失败，请稍后重试');
        } finally {
            followLoading.classList.add('hidden');
        }
    }
    // 渲染关注列表
    function renderFollowList(follows) {
        if (follows.length === 0) {
            noFollow.classList.remove('hidden');
            return;
        }
        noFollow.classList.add('hidden');
        follows.forEach(follow => {
            const followItem = document.createElement('div');
            followItem.className = 'flex items-center justify-between p-3 hover:bg-gray-50 rounded-lg';
            followItem.innerHTML = `
                <div class="flex items-center space-x-3">
                    <img src="${follow.avatar_url}" alt="${follow.username}" class="w-12 h-12 rounded-full object-cover">
                    <div>
                        <h4 class="font-medium">${follow.username}</h4>
                        <p class="text-xs text-gray-500">${follow.school_info}</p>
                    </div>
                </div>
                <button class="px-3 py-1.5 bg-primary text-white rounded-lg text-sm hover:bg-primary-dark transition-colors follow-toggle" data-user-id="${follow.user_id}" data-is-following="true">
                    取消关注
                </button>
            `;
            followList.appendChild(followItem);
        });
    }
    // 绑定关注/取消关注事件
    document.getElementById('follow-list').addEventListener('click', function(e) {
        const btn = e.target.closest('.follow-toggle');
        if (!btn) return;

        const userId = btn.dataset.userId;
        const isFollowing = btn.dataset.isFollowing === 'true';

        if (isFollowing) {
            // 取消关注
            unfollowUser(userId, btn);
        } else {
            // 关注
            followUser(userId, btn);
        }
    });
    // 关注用户
    async function followUser(userId, btn) {
        try {
            btn.disabled = true;
            btn.classList.add('opacity-50', 'cursor-not-allowed');
            btn.innerHTML = '关注中...';
            
            const response = await fetch(`http://localhost:3000/api/follow/${userId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ followerId: globalUser.id })
            });
            
            if (response.ok) {
                btn.dataset.isFollowing = 'true';
                btn.classList.remove('bg-primary-light', 'text-primary');
                btn.classList.add('bg-primary', 'text-white');
                btn.innerHTML = '取消关注';
                alert('关注成功');
            } else {
                throw new Error('关注失败');
            }
        } catch (error) {
            console.error('关注失败:', error);
            alert('关注失败，请稍后重试');
        } finally {
            btn.disabled = false;
            btn.classList.remove('opacity-50', 'cursor-not-allowed');
        }
    }
    // 取消关注用户
    async function unfollowUser(userId, btn) {
        try {
            btn.disabled = true;
            btn.classList.add('opacity-50', 'cursor-not-allowed');
            btn.innerHTML = '取消关注中...';
            
            const response = await fetch(`http://localhost:3000/api/follow/${userId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ followerId: globalUser.id })
            });
            
            if (response.ok) {
                btn.dataset.isFollowing = 'false';
                btn.classList.remove('bg-primary', 'text-white');
                btn.classList.add('bg-primary-light', 'text-primary');
                btn.innerHTML = '关注';
                alert('已取消关注');
                
            } else {
                throw new Error('取消关注失败');
            }
        } catch (error) {
            console.error('取消关注失败:', error);
            alert('取消关注失败，请稍后重试');
        } finally {
            btn.disabled = false;
            btn.classList.remove('opacity-50', 'cursor-not-allowed');
        }
    }

    // -----------------------------点赞模态框----------------------------------
    // 打开点赞列表模态框
    function openLikeModal() {
        if (checkLoginStatusAndPrompt()) return; 
        likeLoading.classList.remove('hidden');
        likeList.innerHTML = '';
        noLike.classList.add('hidden');
        likeModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
        
        // 加载点赞列表
        loadLikeList();
    }
    // 关闭点赞列表模态框
    function closeLikeModalFunc() {
        likeModal.classList.add('hidden');
        document.body.style.overflow = '';
    }
    // 加载点赞列表
    async function loadLikeList() {
        try {
        const response = await fetch(`http://localhost:3000/api/getLikes/${globalUser.id}`);
        if (!response.ok) throw new Error('获取点赞列表失败');
        
        const result = await response.json();
        renderLikeList(result.data);
        } catch (error) {
        console.error('加载点赞列表失败:', error);
        alert('加载点赞列表失败，请稍后重试');
        } finally {
        likeLoading.classList.add('hidden');
        }
    }
    // 渲染点赞列表
    function renderLikeList(likes) {
        if (likes.length === 0) {
        noLike.classList.remove('hidden');
        return;
        }
        noLike.classList.add('hidden');
        
        likes.forEach(like => {
            console.log(like.created_at);
            const likeItem = document.createElement('div');
            likeItem.className = 'flex items-center justify-between p-3 hover:bg-gray-50 rounded-lg';
            likeItem.innerHTML = `
                <div class="flex items-center space-x-3">
                <img src="${like.avatar_url}" alt="${like.username}" class="w-12 h-12 rounded-full object-cover">
                <div>
                    <h4 class="font-medium">${like.username}</h4>
                    <p class="text-xs text-gray-500">${like.school_info}</p>
                    <p class="text-xs text-gray-500">点赞了 ${formatTimeAgo(new Date(like.created_at))}</p>
                </div>
                </div>
                <button class="px-3 py-1.5 bg-white text-primary border border-primary rounded-lg text-sm hover:bg-primary-light transition-colors view-post" data-post-id="${like.post_id}">
                查看动态
                </button>
        `;
        likeList.appendChild(likeItem);
        });

        // 绑定查看动态事件
        document.querySelectorAll('.view-post').forEach(btn => {
            btn.addEventListener('click', function() {
                const postId = this.dataset.postId;
                viewPostDetail(postId);
                closeLikeModalFunc();
            });
        });
    }
    function viewPostDetail(postId){
        alert(`查看动态详情，帖子ID: ${postId}`);
    }

    //------------------------------收藏模态框----------------------------------
    // 打开收藏列表模态框
    function openCollectModal() {
        if (checkLoginStatusAndPrompt()) return; 
        collectLoading.classList.remove('hidden');
        collectList.innerHTML = '';
        noCollect.classList.add('hidden');
        collectModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
        
        // 加载收藏列表
        loadCollectList();
    }
    // 关闭收藏列表模态框
    function closeCollectModalFunc() {
        collectModal.classList.add('hidden');
        document.body.style.overflow = '';
    }
    // 加载收藏列表
    async function loadCollectList() {
        try {
            const response = await fetch(`http://localhost:3000/api/getCollects/${globalUser.id}`);
            if (!response.ok) throw new Error('获取收藏列表失败');
            
            const result = await response.json();
            renderCollectList(result.data);
        } catch (error) {
            console.error('加载收藏列表失败:', error);
            alert('加载收藏列表失败，请稍后重试');
        } finally {
            collectLoading.classList.add('hidden');
        }
    }
    // 渲染收藏列表
    function renderCollectList(collects) {
        if (collects.length === 0) {
            noCollect.classList.remove('hidden');
            return;
        }
        noCollect.classList.add('hidden');
        
        collects.forEach(collect => {
            const collectItem = document.createElement('div');
            collectItem.className = 'border-b border-gray-100 pb-4 last:border-0';
            collectItem.innerHTML = `
                <div class="flex space-x-3 mb-3">
                    <img src="${collect.avatar_url}" alt="${collect.username}" class="w-10 h-10 rounded-full object-cover">
                    <div class="flex-1">
                        <h4 class="font-medium">${collect.username}</h4>
                        <p class="text-xs text-gray-500">${collect.school_info}</p>
                    </div>
                </div>
                <p class="text-gray-700 mb-3">${collect.content}</p>
                ${collect.media_url && collect.media_url.length > 0 
                    ? `<div class="overflow-x-auto pb-3">
                        <div class="flex space-x-2 min-w-max">
                            ${collect.media_url.map(url => `
                                <div class="w-24 h-24 shrink-0 relative cursor-pointer" onclick="openImagePreview('${url}')">
                                    <img src="${url}" alt="收藏图片" class="w-full h-full object-cover rounded">
                                    <div class="absolute inset-0 bg-black/30 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
                                        <i class="fa fa-search-plus text-white text-xl"></i>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>` 
                    : ''}
                <div class="flex items-center justify-between mt-3">
                    <div class="flex items-center space-x-3 text-sm">
                        <span class="text-gray-500">${formatTimeAgo(new Date(collect.created_at))}</span>
                        <button class="text-gray-500 hover:text-primary uncollect-btn" data-post-id="${collect.post_id}">
                            <i class="fa fa-bookmark-o mr-1"></i> 取消收藏
                        </button>
                    </div>
                    <button class="px-3 py-1 bg-primary-light text-primary text-sm rounded-lg hover:bg-primary hover:text-white transition-colors view-collect-post" data-post-id="${collect.post_id}">
                        查看详情
                    </button>
                </div>
            `;
            collectList.appendChild(collectItem);
        });

        // 绑定取消收藏事件
        document.querySelectorAll('.uncollect-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const postId = this.dataset.postId;
                uncollectPost(postId, this);
            });
        });
        // 绑定查看详情事件
        document.querySelectorAll('.view-collect-post').forEach(btn => {
            btn.addEventListener('click', function() {
                const postId = this.dataset.postId;
                viewPostDetail(postId);
                closeCollectModalFunc();
            });
        });
    }
    // 取消收藏帖子
    async function uncollectPost(postId, btn) {
        try {
            btn.disabled = true;
            btn.classList.add('opacity-50', 'cursor-not-allowed');
            btn.innerHTML = '<i class="fa fa-spinner fa-spin mr-1"></i> 取消中...';
            
            const response = await fetch(`http://localhost:3000/api/collect/${postId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ user_id: globalUser.id })
            });
            
            if (response.ok) {
                const result = await response.json();
                // 从UI中移除该收藏项
                const collectItem = btn.closest('.border-b');
                if (collectItem) {
                    collectItem.remove();
                }
                // 检查是否还有收藏项
                if (collectList.children.length === 0) {
                    noCollect.classList.remove('hidden');
                }
                alert('已取消收藏');
            } else {
                throw new Error('取消收藏失败');
            }
        } catch (error) {
            console.error('取消收藏失败:', error);
            alert('取消收藏失败，请稍后重试');
        } finally {
            btn.disabled = false;
            btn.classList.remove('opacity-50', 'cursor-not-allowed');
        }
    }
    function viewPostDetail(postId) {
        alert(`查看帖子详情，帖子ID: ${postId}`);
    }

    // -------------------------------推荐关注---------------------------------
    // 加载推荐关注用户
    function loadRecommendUsers() {
        recommendLoading.classList.remove('hidden');
        recommendUsersContainer.innerHTML = '';
        
        fetch(`http://localhost:3000/api/recommend_users/${globalUser.id}`)
            .then(response => {
                if (!response.ok) throw new Error('获取推荐用户失败');
                    return response.json();
                })
            .then(result => {
                renderRecommendUsers(result.data);
            })
            .catch(error => {
                console.error('加载推荐用户失败:', error);
                noRecommend.classList.remove('hidden');
                noRecommend.textContent = '加载推荐失败，请稍后重试';
            })
            .finally(() => {
                recommendLoading.classList.add('hidden');
            });
    }
    // 渲染推荐用户
    function renderRecommendUsers(users) {
        // 过滤已关注的用户并限制最多显示3个
        const filteredUsers = users
            .filter(user => !user.is_following)
            .slice(0, 3);
            
        if (filteredUsers.length === 0) {
            noRecommend.classList.remove('hidden');
            noRecommend.innerHTML = `
            <i class="fa fa-user-o text-2xl mb-2"></i>
            <p>没有适合的推荐用户</p>
            `;
            return;
        }
        
        noRecommend.classList.add('hidden');
        
        filteredUsers.forEach(user => {
            const userItem = document.createElement('div');
            userItem.className = 'flex items-center space-x-3';
            userItem.innerHTML = `
            <a href="Personal.html?userId=${user.user_id}&LoginUserId=${globalUser.id}&LoginUserName=${globalUser.username}&LoginAvatarURL=${globalUser.avatar_url}" class="flex items-center space-x-3">
                <img src="${user.avatar_url}" alt="${user.username}" class="w-10 h-10 rounded-full object-cover">
            </a>
            <div class="flex-1">
                <h4 class="font-medium text-sm">${user.username}</h4>
                <p class="text-xs text-gray-500">${user.school_info}</p>
            </div>
            <button class="px-3 py-1 bg-primary-light text-primary text-xs rounded-lg hover:bg-primary hover:text-white transition-colors follow-btn" data-user-id="${user.user_id}">
                关注
            </button>
            `;
            recommendUsersContainer.appendChild(userItem);
        });
        
        // 绑定关注按钮事件
        document.querySelectorAll('.follow-btn').forEach(btn => {
            btn.addEventListener('click', function() {
            const userId = this.dataset.userId;
            followRecommendedUser(userId, this);
            });
        });
    }
    // 关注推荐用户
    function followRecommendedUser(userId, btn) {
        if (checkLoginStatusAndPrompt()) return; 
        btn.disabled = true;
        btn.classList.add('opacity-50', 'cursor-not-allowed');
        btn.innerHTML = '<i class="fa fa-spinner fa-spin mr-1"></i> 关注中...';
        
        fetch(`http://localhost:3000/api/follow/${userId}`, {
            method: 'POST',
            headers: {
            'Content-Type': 'application/json'
            },
            body: JSON.stringify({ followerId: globalUser.id })
        })
        .then(response => {
            if (!response.ok) throw new Error('关注失败');
            return response.json();
        })
        .then(() => {
            btn.dataset.isFollowing = 'true';
            btn.classList.remove('bg-primary-light', 'text-primary');
            btn.classList.add('bg-success', 'text-white');
            btn.innerHTML = '已关注';
            alert('关注成功');
            
            // 从推荐列表中移除该用户
            const userItem = btn.closest('.flex');
            if (userItem) {
            userItem.remove();
            }
            
            // 检查是否还有推荐用户
            if (recommendUsersContainer.children.length === 0) {
            noRecommend.classList.remove('hidden');
            noRecommend.innerHTML = `
                <i class="fa fa-user-o text-2xl mb-2"></i>
                <p>没有更多推荐用户</p>
            `;
            }
        })
        .catch(error => {
            console.error('关注推荐用户失败:', error);
            alert('关注失败，请稍后重试');
        })
        .finally(() => {
            btn.disabled = false;
            btn.classList.remove('opacity-50', 'cursor-not-allowed');
        });
    }

    //--------------------------------主题切换--------------------------------
    const themeToggle = document.getElementById('theme-toggle');
    const htmlElement = document.documentElement;
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark' || (savedTheme === null && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        switchToDarkTheme();
    } else {
        switchToLightTheme();
    }
    // 切换到黑暗主题
    function switchToDarkTheme() {
        htmlElement.classList.remove('light');
        htmlElement.classList.add('dark');
        document.body.classList.remove('bg-gray-50', 'text-dark');
        document.body.classList.add('bg-dark-bg', 'text-dark-text');
        // 切换导航栏背景
        const header = document.querySelector('header');
        header.classList.remove('bg-white');
        header.classList.add('bg-dark-card', 'border-dark-border');
        // 切换卡片背景
        const cards = document.querySelectorAll('.bg-white');
            cards.forEach(card => {
            card.classList.remove('bg-white', 'border-gray-100');
            card.classList.add('bg-dark-card', 'border-dark-border');
        });
        // 切换图标颜色
        const icons = document.querySelectorAll('i');
        icons.forEach(icon => {
            if (icon.classList.contains('text-gray-400')) {
                icon.classList.remove('text-gray-400');
                icon.classList.add('text-gray-500');
            }   
            if (icon.classList.contains('text-gray-500')) {
                icon.classList.remove('text-gray-500');
                icon.classList.add('text-gray-400');
            }
            if (icon.classList.contains('text-gray-600')) {
                icon.classList.remove('text-gray-600');
                icon.classList.add('text-gray-300');
            }
        });
        // 切换按钮图标
        themeToggle.innerHTML = '<i class="fa fa-sun-o text-yellow-400"></i>';
        // 保存主题偏好
        localStorage.setItem('theme', 'dark');
    }
    // 切换到浅色主题
    function switchToLightTheme() {
        htmlElement.classList.remove('dark');
        htmlElement.classList.add('light');
        document.body.classList.remove('bg-dark-bg', 'text-dark-text');
        document.body.classList.add('bg-gray-50', 'text-dark');
        // 切换导航栏背景
        const header = document.querySelector('header');
        header.classList.remove('bg-dark-card', 'border-dark-border');
        header.classList.add('bg-white');
        // 切换卡片背景
        const cards = document.querySelectorAll('.bg-dark-card');
        cards.forEach(card => {
            card.classList.remove('bg-dark-card', 'border-dark-border');
            card.classList.add('bg-white', 'border-gray-100');
        });
        // 切换图标颜色
        const icons = document.querySelectorAll('i');
        icons.forEach(icon => {
            if (icon.classList.contains('text-gray-500')) {
                icon.classList.remove('text-gray-500');
                icon.classList.add('text-gray-400');
            }
            if (icon.classList.contains('text-gray-400')) {
                icon.classList.remove('text-gray-400');
                icon.classList.add('text-gray-500');
                }
            if (icon.classList.contains('text-gray-300')) {
                icon.classList.remove('text-gray-300');
                icon.classList.add('text-gray-600');
            }
        });
        // 切换按钮图标
        themeToggle.innerHTML = '<i class="fa fa-moon-o text-gray-600"></i>';
        // 保存主题偏好
        localStorage.setItem('theme', 'light');
    }
    // 主题切换按钮点击事件
    themeToggle.addEventListener('click', () => {
        if (htmlElement.classList.contains('dark')) {
            switchToLightTheme();
        } else {
            switchToDarkTheme();
        }
    });

    //------------------------打卡------------------------
    const calendarDays = document.getElementById('calendar-days');
    const calendarTitle = document.getElementById('calendar-title');
    const currentMonthYear = document.getElementById('current-month-year');
    const prevMonthBtn = document.getElementById('prev-month');
    const nextMonthBtn = document.getElementById('next-month');
    const checkinBtn = document.getElementById('checkin-btn');
    const checkinStatus = document.getElementById('checkin-status');
    const monthCheckinCount = document.getElementById('month-checkin-count');
    const streakCount = document.getElementById('streak-count');
    if (!calendarDays) return; // 如果元素不存在，不执行后续代码
    let currentDate = new Date();
    let currentMonth = currentDate.getMonth();
    let currentYear = currentDate.getFullYear();
    // 初始化日历
    initCalendar();
    // 获取当前月份的打卡记录
    loadMonthCheckins(currentYear, currentMonth);
    // 上个月按钮点击事件
    prevMonthBtn.addEventListener('click', function () {
        currentMonth--;
        if (currentMonth < 0) {
            currentMonth = 11;
            currentYear--;
        }
        renderCalendar(currentYear, currentMonth);
        loadMonthCheckins(currentYear, currentMonth);
    });
    // 下个月按钮点击事件
    nextMonthBtn.addEventListener('click', function () {
        currentMonth++;
        if (currentMonth > 11) {
            currentMonth = 0;
            currentYear++;
        }
        renderCalendar(currentYear, currentMonth);
        loadMonthCheckins(currentYear, currentMonth);
    });
    // 打卡按钮点击事件
    checkinBtn.addEventListener('click', function () {
        const today = new Date();
        today.setHours(0, 0, 0, 0);

        // 如果不是当前月份，不允许打卡
        if (currentYear !== today.getFullYear() || currentMonth !== today.getMonth()) {
            alert('只能在当前月份进行打卡！');
            return;
        }

        // 发送打卡请求
        checkin();
    });
    // 初始化日历
    function initCalendar() {
        renderCalendar(currentYear, currentMonth);
        updateCheckinStatus();
    }
    // 渲染日历
    function renderCalendar(year, month) {
        calendarDays.innerHTML = '';

        const firstDay = new Date(year, month, 1);
        const lastDay = new Date(year, month + 1, 0);
        const daysInMonth = lastDay.getDate();
        const startingDay = firstDay.getDay(); // 0 = Sunday

        // 更新标题
        const monthNames = ['一月', '二月', '三月', '四月', '五月', '六月', '七月', '八月', '九月', '十月', '十一月', '十二月'];
        calendarTitle.textContent = `${year}年 ${monthNames[month]}`;
        currentMonthYear.textContent = `${year}年${month + 1}月`;

        // 添加空白格子（上个月的日期）
        for (let i = 0; i < startingDay; i++) {
            const emptyDay = document.createElement('div');
            emptyDay.className = 'text-gray-300';
            calendarDays.appendChild(emptyDay);
        }

        // 添加当月日期
        const today = new Date();
        today.setHours(0, 0, 0, 0);

        for (let i = 1; i <= daysInMonth; i++) {
            const dayElement = document.createElement('div');
            const currentDateObj = new Date(year, month, i);

            // 默认样式
            dayElement.className = 'text-center py-1 cursor-pointer hover:bg-gray-200 rounded';
            dayElement.textContent = i;
            dayElement.dataset.date = `${year}-${String(month + 1).padStart(2, '0')}-${String(i).padStart(2, '0')}`;

            // 如果是今天，添加不同样式
            if (currentDateObj.getTime() === today.getTime()) {
                dayElement.classList.add('bg-primary', 'text-white');
            }

            // 如果是未来日期，设置为不可点击
            if (currentDateObj > today) {
                dayElement.classList.remove('cursor-pointer', 'hover:bg-gray-200');
                dayElement.classList.add('text-gray-400', 'cursor-not-allowed');
            } else {
                // 为过去的日期添加点击事件
                dayElement.addEventListener('click', function () {
                    const dateStr = this.dataset.date;
                    showCheckinDetails(dateStr);
                });
            }

            calendarDays.appendChild(dayElement);
        }
    }
    // 获取某月的打卡记录
    async function loadMonthCheckins(year, month) {
        try {
            const response = await fetch(`http://localhost:3000/api/checkins?userId=${globalUser.id}&year=${year}&month=${month + 1}`);

            if (!response.ok) throw new Error('获取打卡记录失败');

            const result = await response.json();

            // 更新日历上的打卡状态
            updateCalendarCheckins(result.data);

            // 更新打卡统计
            monthCheckinCount.textContent = result.data.length;
            streakCount.textContent = result.streakCount || 0;

        } catch (error) {
            console.error('加载打卡记录失败:', error);
        }
    }
    // 更新日历上的打卡状态
    function updateCalendarCheckins(checkins) {
        // 重置所有日期的打卡状态
        document.querySelectorAll('#calendar-days > div').forEach(day => {
            day.classList.remove('bg-success', 'text-white', 'border-success');

            // 保留今天的样式
            const today = new Date();
            today.setHours(0, 0, 0, 0);
            const dayDate = day.dataset.date ? new Date(day.dataset.date) : null;

            if (dayDate && dayDate.getTime() === today.getTime()) {
                day.classList.add('bg-primary', 'text-white');
            }
        });

        // 为已打卡的日期添加样式
        checkins.forEach(checkin => {
            // 直接使用服务器返回的日期，不做时区调整
            let dateStr = checkin.checkin_date.split('T')[0];

            // 使用原始日期查找元素
            const dayElement = document.querySelector(`#calendar-days > div[data-date="${dateStr}"]`);

            if (dayElement) {
                // 如果是今天，保持今天的样式，但添加边框表示已打卡
                if (dayElement.classList.contains('bg-primary')) {
                    dayElement.classList.add('border-2', 'border-success');
                } else {
                    // 否则使用打卡样式
                    dayElement.classList.remove('hover:bg-gray-200');
                    dayElement.classList.add('bg-success', 'text-white');
                }
            }
        });
        // 更新打卡按钮状态
        updateCheckinButtonState(checkins);
        document.querySelectorAll('#calendar-days > div[data-date]').forEach(day => {
        });

        // 为每个打卡记录找到对应的日期元素并显示
        checkins.forEach(checkin => {
            const dateStr = checkin.checkin_date.split('T')[0];

        });
    }
    // 更新打卡按钮状态
    function updateCheckinButtonState(checkins) {
        const today = new Date();
        const year = today.getFullYear();
        const month = String(today.getMonth() + 1).padStart(2, '0');
        const day = String(today.getDate()).padStart(2, '0');
        const todayStr = `${year}-${month}-${day}`;

        // 更清晰的比较逻辑，打印出日期便于调试
        console.log('今日日期:', todayStr);
        const hasCheckedInToday = checkins.some(checkin => {
            const checkinDate = checkin.checkin_date.split('T')[0];
            console.log('检查打卡日期:', checkinDate);
            return checkinDate === todayStr;
        });

        if (hasCheckedInToday) {
            checkinBtn.disabled = true;
            checkinBtn.classList.remove('bg-primary', 'hover:bg-primary-dark');
            checkinBtn.classList.add('bg-gray-300', 'cursor-not-allowed');
            checkinBtn.innerHTML = '<i class="fa fa-check mr-1"></i> 今日已打卡';

            checkinStatus.textContent = '✅ 太棒了！今天已经完成打卡';
            checkinStatus.classList.add('text-success');
        } else {
            checkinBtn.disabled = false;
            checkinBtn.classList.remove('bg-gray-300', 'cursor-not-allowed');
            checkinBtn.classList.add('bg-primary', 'hover:bg-primary-dark');
            checkinBtn.innerHTML = '<i class="fa fa-check-circle mr-1"></i> 今日打卡';

            checkinStatus.textContent = '📅 今天还没有打卡哦';
            checkinStatus.classList.remove('text-success');
        }

        // 如果不是当前月，禁用打卡按钮
        const currentViewMonth = currentMonth;
        const currentViewYear = currentYear;

        if (currentViewYear !== today.getFullYear() || currentViewMonth !== today.getMonth()) {
            checkinBtn.disabled = true;
            checkinBtn.classList.remove('bg-primary', 'hover:bg-primary-dark');
            checkinBtn.classList.add('bg-gray-300', 'cursor-not-allowed');
            checkinStatus.textContent = '只能在当前月份进行打卡';
        }
    }
    // 显示打卡详情
    function showCheckinDetails(dateStr) {
        // 加载该日期的打卡记录
        fetch(`http://localhost:3000/api/checkins/date?userId=${globalUser.id}&date=${dateStr}`)
            .then(response => {
                if (!response.ok) throw new Error('获取打卡详情失败');
                return response.json();
            })
            .then(result => {
                if (result.data) {
                    alert(`${dateStr} 打卡成功！${result.data.comment ? '\n备注: ' + result.data.comment : ''}`);
                } else {
                    alert(`${dateStr} 没有打卡记录`);
                }
            })
            .catch(error => {
                console.error('获取打卡详情失败:', error);
                alert('获取打卡详情失败，请稍后重试');
            });
    }
    // 修改 load.js 中的 checkin() 函数
    async function checkin() {
        try {
            const comment = prompt('请输入今日打卡心情或备注（可选）:');

            // 获取本地时区的今天日期 (修改这部分，不使用toISOString)
            const today = new Date();
            const year = today.getFullYear();
            const month = String(today.getMonth() + 1).padStart(2, '0');
            const day = String(today.getDate()).padStart(2, '0');
            const clientDate = `${year}-${month}-${day}`;

            const response = await fetch(`http://localhost:3000/api/checkin`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    userId: globalUser.id,
                    comment: comment || '',
                    clientDate: clientDate  // 本地日期格式
                })
            });

            if (!response.ok) throw new Error('打卡失败');

            const result = await response.json();

            if (result.success) {
                alert('打卡成功！');
                // 直接更新UI
                checkinBtn.disabled = true;
                checkinBtn.classList.remove('bg-primary', 'hover:bg-primary-dark');
                checkinBtn.classList.add('bg-gray-300', 'cursor-not-allowed');
                checkinBtn.innerHTML = '<i class="fa fa-check mr-1"></i> 今日已打卡';
                checkinStatus.textContent = '✅ 太棒了！今天已经完成打卡';
                checkinStatus.classList.add('text-success');

                // 然后再重新加载当月打卡记录
                loadMonthCheckins(currentYear, currentMonth);
            } else {
                throw new Error(result.message || '打卡失败');
            }
        } catch (error) {
            console.error('打卡失败:', error);
            alert(error.message || '打卡失败，请稍后重试');
        }
    }
    // 更新打卡状态
    function updateCheckinStatus() {
        const today = new Date();
        const todayStr = today.toISOString().split('T')[0];

        fetch(`http://localhost:3000/api/checkins/date?userId=${globalUser.id}&date=${todayStr}`)
            .then(response => {
                if (!response.ok) throw new Error('获取打卡状态失败');
                return response.json();
            })
            .then(result => {
                if (result.data) {
                    checkinBtn.disabled = true;
                    checkinBtn.classList.remove('bg-primary', 'hover:bg-primary-dark');
                    checkinBtn.classList.add('bg-gray-300', 'cursor-not-allowed');
                    checkinBtn.innerHTML = '<i class="fa fa-check mr-1"></i> 今日已打卡';

                    checkinStatus.textContent = '✅ 太棒了！今天已经完成打卡';
                    checkinStatus.classList.add('text-success');
                }
            })
            .catch(error => {
                console.error('获取打卡状态失败:', error);
            });
    }

    // -------------------------------搜索--------------------------------
    const mainSearchInput = document.getElementById('main-search-input');
    const mainSearchButton = document.getElementById('main-search-button');
    const searchResultsModal = document.getElementById('search-results-modal');
    const closeSearchResultsModalBtn = document.getElementById('close-search-results-modal');
    const searchLoadingIndicator = document.getElementById('search-loading-indicator');
    const noSearchResultsMessage = document.getElementById('no-search-results');
    const searchedTermDisplay = document.getElementById('searched-term-display');
    const userResultsContainer = document.getElementById('user-results-container');
    const userResultsList = document.getElementById('user-results-list');
    const postResultsContainer = document.getElementById('post-results-container');
    const postResultsList = document.getElementById('post-results-list');
    // 打开搜索模态框
    function openSearchResultsModal() {
        searchResultsModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden'; // 防止背景滚动
    }
    // 关闭搜索模态框
    function closeSearchResultsModal() {
        searchResultsModal.classList.add('hidden');
        document.body.style.overflow = '';
    }
    // 清理并重置搜索模态框状态
    function resetSearchModal() {
        searchLoadingIndicator.classList.add('hidden');
        noSearchResultsMessage.classList.add('hidden');
        userResultsContainer.classList.add('hidden');
        postResultsContainer.classList.add('hidden');
        userResultsList.innerHTML = '';
        postResultsList.innerHTML = '';
    }
    // 执行搜索
    async function performSearch() {
        const searchTerm = mainSearchInput.value.trim();
        if (!searchTerm) {
            alert('请输入搜索内容！');
            return;
        }
        openSearchResultsModal();
        resetSearchModal();
        searchLoadingIndicator.classList.remove('hidden');
        try {
            const response = await fetch(`http://localhost:3000/api/search?query=${encodeURIComponent(searchTerm)}&userId=${globalUser.id}`);
            if (!response.ok) {
                throw new Error('搜索请求失败');
            }
            const results = await response.json();
            searchLoadingIndicator.classList.add('hidden');
            if (results.users.length === 0 && results.posts.length === 0) {
                searchedTermDisplay.textContent = searchTerm;
                noSearchResultsMessage.classList.remove('hidden');
            } else {
                renderUserResults(results.users);
                renderPostResults(results.posts);
            }

        } catch (error) {
            console.error('搜索失败:', error);
            searchLoadingIndicator.classList.add('hidden');
            searchedTermDisplay.textContent = searchTerm;
            noSearchResultsMessage.classList.remove('hidden');
            noSearchResultsMessage.querySelector('p').innerHTML += '<br>搜索出错，请稍后重试。'; // 额外提示
        }
    }
    // 渲染用户搜索结果
    function renderUserResults(users) {
        if (users.length === 0) {
            userResultsContainer.classList.add('hidden');
            return;
        }
        userResultsList.innerHTML = ''; // 清空旧结果
        users.forEach(user => {
            const userItem = document.createElement('div');
            userItem.className = 'flex items-center p-3 hover:bg-gray-100 rounded-lg transition-colors';
            // 注意 personal.html 的链接格式，确保它能正确处理 userId
            userItem.innerHTML = `
                <a href="personal.html?userId=${user.user_id}&LoginUserId=${globalUser.id}&LoginUserName=${globalUser.username}&LoginAvatarURL=${globalUser.avatar_url}" class="flex items-center w-full">
                    <img src="${user.avatar_url}" alt="${user.username}" class="w-10 h-10 rounded-full object-cover mr-3">
                    <div>
                        <h5 class="font-medium text-sm">${user.username}</h5>
                        <p class="text-xs text-gray-500">${user.gender}</p>
                    </div>
                </a>
            `;
            userResultsList.appendChild(userItem);
        });
        userResultsContainer.classList.remove('hidden');
    }
    // 渲染动态搜索结果
    function renderPostResults(posts) {
        if (posts.length === 0) {
            postResultsContainer.classList.add('hidden');
            return;
        }
        postResultsList.innerHTML = ''; // 清空旧结果
        posts.forEach(post => {
            const postItem = document.createElement('div');
            postItem.className = 'p-3 border-b border-gray-200 last:border-b-0 hover:bg-gray-50 rounded-md';

            // 截断帖子内容，显示部分预览
            const contentSnippet = post.content.length > 100 ? post.content.substring(0, 100) + '...' : post.content;
            // 简单处理媒体URL，假设是JSON字符串数组
            let mediaPreviewHtml = '';
            if (post.media_url) {
                try {
                    const mediaUrls = JSON.parse(post.media_url);
                    if (Array.isArray(mediaUrls) && mediaUrls.length > 0) {
                        // 只显示第一张作为预览
                        const firstImageUrl = mediaUrls[0].startsWith('http') ? mediaUrls[0] : `http://localhost:3000${mediaUrls[0]}`;
                        mediaPreviewHtml = `<img src="${firstImageUrl}" alt="动态图片" class="w-16 h-16 object-cover rounded-md mt-2">`;
                    }
                } catch (e) { /* 解析失败则不显示图片 */ }
            }
            postItem.innerHTML = `
                <div class="flex items-start space-x-2">
                    <a href="personal.html?userId=${post.author_user_id}?LoginUserId=${globalUser.id}&LoginUserName=${globalUser.username}&LoginAvatarURL=${globalUser.avatar_url}" class="flex-shrink-0">
                    <img src="${post.author_avatar_url}" alt="${post.author_username}" class="w-8 h-8 rounded-full object-cover">
                    </a>
                    <div>
                        <a href="personal.html?userId=${post.author_user_id}" class="font-medium text-xs text-gray-700 hover:underline">${post.author_username}</a>
                        <span class="text-xs text-gray-400 ml-1">${formatTimeAgo(post.created_at)}</span>
                        <p class="text-sm text-gray-800 mt-1">${contentSnippet}</p>
                        ${mediaPreviewHtml}
                    </div>
                </div>
            `;
            postResultsList.appendChild(postItem);
        });
        postResultsContainer.classList.remove('hidden');
    }
    if (mainSearchButton) {
        mainSearchButton.addEventListener('click', performSearch);
    }
    if (mainSearchInput) {
        mainSearchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                performSearch();
            }
        });
    }
    if (closeSearchResultsModalBtn) {
        closeSearchResultsModalBtn.addEventListener('click', closeSearchResultsModal);
    }
    // 点击模态框外部关闭 (可选)
    if (searchResultsModal) {
        searchResultsModal.addEventListener('click', function (e) {
            if (e.target === searchResultsModal) {
                closeSearchResultsModal();
            }
        });
    }


});
    
// 内联事件处理程序需要全局作用域
function openImagePreview(url) {
        // 创建预览模态框
        const previewModal = document.createElement('div');
        previewModal.className = 'fixed inset-0 z-50 flex items-center justify-center bg-black/90';
        previewModal.innerHTML = `
            <button class="absolute top-4 right-4 text-white text-3xl" onclick="document.body.removeChild(this.parentElement)">
                <i class="fa fa-times"></i>
            </button>
            <img src="${url}" alt="图片预览" class="max-w-4xl max-h-[90vh] object-contain">
        `;
        document.body.appendChild(previewModal);
    }

// 格式化时间为相对时间
function formatTimeAgo(date) {
    // 转换为本地时间（中国标准时间）
    const localDate = new Date(date.toLocaleString('zh-CN', {
        timeZone: 'Asia/Shanghai'
    }));
    
    const now = new Date();
    const diffMs = now - localDate;
    const diffSeconds = Math.floor(diffMs / 1000);
    const diffMinutes = Math.floor(diffSeconds / 60);
    const diffHours = Math.floor(diffMinutes / 60);
    const diffDays = Math.floor(diffHours / 24);
    
    if (diffSeconds < 10) {
        return '刚刚';
    } else if (diffSeconds < 60) {
        return `${diffSeconds}秒前`;
    } else if (diffMinutes < 60) {
        return `${diffMinutes}分钟前`;
    } else if (diffHours < 24) {
        return `${diffHours}小时前`;
    } else if (diffDays < 3) {
        return `${diffDays}天前`;
    } else if (diffDays < 30) {
        return `${diffDays}天前`;
    } else {
        return localDate.toLocaleDateString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit'
        });
    }
}

//-------------------------------私信--------------------------------
// 定义好友及聊天记录数据
const friends = [
    {
        id: 1,
        avatar: 'https://picsum.photos/id/64/100/100', // 温奇炜头像
        name: '温奇炜、初旭 (3)',
        lastMsg: '你的个人主页personnal先不动',
        messages: [
            { sender: 'them', time: '19:52', content: '你的个人主页personnal先不动' },
            { sender: 'me', time: '19:54', content: '好的，我先处理其他部分' },
            { sender: 'them', time: '19:56', content: '我那个还没改完' },
            { sender: 'them', time: '20:04', content: '加的话不用 到时候交接的时候附加一下 说明' },
            { sender: 'me', time: '20:06', content: '明白，我会注意的' }
        ]
    },
    {
        id: 2,
        avatar: 'https://picsum.photos/id/22/100/100', // 郑郑郑头像
        name: '郑郑郑在向前冲',
        lastMsg: '你撤回了一条消息',
        messages: [
            { sender: 'them', time: '15:30', content: '在吗？' },
            { sender: 'me', time: '15:31', content: '在的~' },
            { sender: 'them', time: '15:32', content: '刚才发错了，撤回一条消息' },
            { sender: 'me', time: '15:33', content: '哈哈，没事' }
        ]
    },
    {
        id: 3,
        avatar: 'https://picsum.photos/id/24/100/100', // 联想头像
        name: '联想',
        lastMsg: 'RTX™5090独显+192G+...',
        messages: [
            { sender: 'them', time: '10:00', content: '新显卡到了，RTX™5090独显+192G内存' },
            { sender: 'me', time: '10:01', content: '这么强！' },
            { sender: 'them', time: '10:02', content: '是的，测试一下性能' }
        ]
    }
];
// 渲染好友列表
function renderFriendList() {
    const friendList = document.getElementById('friend-list');
    friendList.innerHTML = ''; // 清空原有内容

    friends.forEach(friend => {
        const item = document.createElement('div');
        item.className = 'message-friend-item';
        item.dataset.id = friend.id; // 绑定好友ID，用于切换聊天
        item.innerHTML = `
            <div class="p-4 flex items-center space-x-3">
            <img src="${friend.avatar}" alt="${friend.name}" class="w-10 h-10 rounded-full object-cover">
            <div>
                <h4 class="font-semibold">${friend.name}</h4>
                <p class="text-xs text-gray-500">${friend.lastMsg}</p>
            </div>
            </div>
        `;

        // 绑定点击事件：切换聊天
        item.addEventListener('click', () => switchChat(friend.id));
        friendList.appendChild(item);
    });

    // 默认选中第一个好友
    if (friends.length > 0) {
        document.querySelector('.message-friend-item').classList.add('active');
        switchChat(friends[0].id);
    }
}
//切换聊天
function switchChat(friendId) {
    // 1. 更新好友列表的选中状态
    document.querySelectorAll('.message-friend-item').forEach(item => {
        item.classList.remove('active');
    });
    document.querySelector(`[data-id="${friendId}"]`).classList.add('active');

    // 2. 获取当前好友数据
    const friend = friends.find(f => f.id === friendId);
    if (!friend) return;

    // 3. 更新聊天头部（头像 + 名称）
    const chatHeader = document.getElementById('chat-header');
    const chatTitle = document.getElementById('chat-title');
    chatHeader.querySelector('img').src = friend.avatar;
    chatTitle.textContent = friend.name;

    // 4. 渲染聊天记录
    const messageContent = document.getElementById('message-content');
    messageContent.innerHTML = ''; // 清空原有内容

    friend.messages.forEach(msg => {
        const msgDiv = document.createElement('div');

        if (msg.sender === 'them') { // 对方消息：左对齐
            msgDiv.className = 'flex items-start space-y-1 mb-4';
            msgDiv.innerHTML = `
            <img src="${friend.avatar}" alt="对方头像" class="w-8 h-8 rounded-full object-cover mr-2">
            <div class="bg-gray-100 rounded-lg rounded-tl-none px-3 py-2 max-w-[70%]">
                <p class="text-sm text-gray-500">${friend.name.split('、')[0]} ${msg.time}</p>
                <p>${msg.content}</p>
            </div>
            `;
        } else { // 自己消息：右对齐
            msgDiv.className = 'flex items-start justify-end space-y-1 mb-4';
            msgDiv.innerHTML = `
            <div class="bg-primary text-white rounded-lg rounded-tr-none px-3 py-2 max-w-[70%]">
                <p class="text-sm text-gray-500">我 ${msg.time}</p>
                <p>${msg.content}</p>
            </div>
            `;
        }
        messageContent.appendChild(msgDiv);
    });

    // 5. 滚动到底部，显示最新消息
    messageContent.scrollTop = messageContent.scrollHeight;
}
//初始化 + 交互逻辑
window.addEventListener('DOMContentLoaded', () => {
    renderFriendList(); // 渲染好友列表

    // 打开私信模态框（绑定到左侧“私信”链接）
    const messageLink = document.querySelector('a:has(.fa-users)');
    const messageModal = document.getElementById('message-modal');
    messageLink.addEventListener('click', (e) => {
        e.preventDefault();
        if (checkLoginStatusAndPrompt()) return; 
        messageModal.classList.remove('hidden');
        // 确保默认选中第一个好友（如果需要）
        if (friends.length > 0) switchChat(friends[0].id);
    });

    // 关闭私信模态框
    const closeMessageModal = document.getElementById('close-message-modal');
    closeMessageModal.addEventListener('click', () => {
        messageModal.classList.add('hidden');
    });
    messageModal.addEventListener('click', (e) => {
        if (e.target === messageModal) messageModal.classList.add('hidden');
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !messageModal.classList.contains('hidden')) {
            messageModal.classList.add('hidden');
        }
    });
});


//---------------------------------登陆注册函数------------------------
// 计算密码强度
function calculatePasswordStrength(password) {
    let strength = 0;
    if (password.length >= 8) strength++;
    if (/[a-z]/.test(password)) strength++;
    if (/[A-Z]/.test(password)) strength++;
    if (/[0-9]/.test(password)) strength++;
    if (/[^a-zA-Z0-9]/.test(password)) strength++;
    return strength;
}
// 更新密码强度显示
function updatePasswordStrengthDisplay(input, strength) {
    const strengthText = document.getElementById('passwordStrengthText');
    const passwordLength = document.getElementById('passwordLength');
    const strengthBars = [
        document.getElementById('strengthBar1'),
        document.getElementById('strengthBar2'),
        document.getElementById('strengthBar3'),
        document.getElementById('strengthBar4')
    ];
    passwordLength.textContent = `${input.value.length}/8`;
    switch (strength) {
        case 0:
        case 1:
            strengthText.textContent = '弱';
            strengthBars.forEach((bar, index) => {
                bar.style.backgroundColor = index === 0 ? 'red' : 'gray';
            });
            break;
        case 2:
            strengthText.textContent = '中';
            strengthBars.forEach((bar, index) => {
                bar.style.backgroundColor = index < 2 ? 'orange' : 'gray';
            });
            break;
        case 3:
        case 4:
            strengthText.textContent = '强';
            strengthBars.forEach((bar, index) => {
                bar.style.backgroundColor = index < 3 ? 'yellowgreen' : 'gray';
            });
            break;
        case 5:
            strengthText.textContent = '非常强';
            strengthBars.forEach(bar => {
                bar.style.backgroundColor = 'green';
            });
            break;
    }
}
// 监听注册密码输入事件
document.addEventListener('DOMContentLoaded', function() {
    const registerPasswordInput = document.getElementById('registerPassword');
    if (registerPasswordInput) {
        registerPasswordInput.addEventListener('input', function() {
            const password = this.value;
            const strength = calculatePasswordStrength(password);
            updatePasswordStrengthDisplay(this, strength);
        });
    }
});
// 切换密码可见性
function togglePasswordVisibility(inputId, button) {
    const passwordInput = document.getElementById(inputId);
    const icon = button.querySelector('i');

    if (passwordInput.type === 'password') {
        passwordInput.type = 'text';
        icon.classList.remove('fa-eye-slash');
        icon.classList.add('fa-eye');
    } else {
        passwordInput.type = 'password';
        icon.classList.remove('fa-eye');
        icon.classList.add('fa-eye-slash');
    }
}


// -------------------------------登录---------------------------------------
document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', function(event) {
            event.preventDefault(); // 阻止表单默认提交行为
            handleLogin();
        });
    }
});

async function handleLogin() {
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    const UserImage = document.querySelectorAll('.login-required');
    // 显示加载状态
    const loginBtn = document.querySelector('#loginForm button[type="submit"]');
    const originalText = loginBtn.innerHTML;
    loginBtn.disabled = true;
    loginBtn.innerHTML = '<i class="fa fa-spinner fa-spin mr-1"></i> 登录中...';
    
    try {
        // 实际应用中调用后端API
        const response = await fetch('http://localhost:3000/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });
        
        const data = await response.json();
        if (response.ok) {
            // 更新全局用户状态
            globalUser.id = data.user.user_id;
            globalUser.token = '模拟令牌_' + new Date().getTime(); // 实际应使用API返回的token
            globalUser.username = data.user.username;
            globalUser.avatar_url = data.user.avatar_url;
            globalUser.type = data.user.type || 0; // 默认普通用户类型
            globalUser.post_count = data.user.post_count;
            globalUser.following_count = data.user.following_count;
            globalUser.follower_count = data.user.follower_count;
            globalUser.school_info = data.user.school_info || ""; // 学校信息
            // 保存登录状态和用户信息
            localStorage.setItem('authToken', globalUser.token);
            localStorage.setItem('userInfo', JSON.stringify({
                userId: data.user.user_id,
                username: data.user.username,
                avatar_url: data.user.avatar_url,
                type: data.user.type || 0, // 默认普通用户类型
                postCount: data.user.post_count,
                followCount: data.user.following_count,
                followerCount: data.user.follower_count,
                school_info: data.user.school_info || "" // 学校信息
            }));
            
            // 关闭模态框
            document.getElementById('login-register-modal').classList.add('hidden');
            
            // 更新界面显示
            updateProfileDisplay();
            updateNavigation(); // 新增：更新导航栏显示
            
            // 提示登录成功
            alert('登录成功！');
            } else {
            throw new Error(data.error || '登录失败，请检查邮箱和密码');
            }
    } catch (error) {
        console.error('登录失败:', error);
        alert(error.message);
    } finally {
        // 恢复登录按钮状态
        loginBtn.disabled = false;
        loginBtn.innerHTML = originalText;
    }
}

// 退出登录功能
document.addEventListener('DOMContentLoaded', function() {
  const logoutBtn = document.getElementById('logout-btn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', function(e) {
      e.preventDefault();
      logout();
    });
  }
  // 用户菜单切换
  const userMenuBtn = document.getElementById('user-menu-btn');
  const userMenu = document.getElementById('user-menu');
  if (userMenuBtn && userMenu) {
    userMenuBtn.addEventListener('click', function() {
      userMenu.classList.toggle('hidden');
    });
    
    // 点击其他地方关闭菜单
    document.addEventListener('click', function(e) {
      if (!userMenuBtn.contains(e.target) && !userMenu.contains(e.target)) {
        userMenu.classList.add('hidden');
      }
    });
  }
});

// 退出登录
function logout() {
  // 清除本地存储
  localStorage.removeItem('authToken');
  localStorage.removeItem('userInfo');
  
  // 重置全局用户状态
  globalUser = {
    id: null,
    token: null,
    refreshToken: null,
    username: "",
    type: 0,
    avatar_url: ""
  };
  
  // 更新界面
  updateProfileDisplay();
  updateNavigation();
  
  alert('已退出登录');
}

// 检查登录状态
function checkLoginStatus() {
  const authToken = localStorage.getItem('authToken');
  const savedUser = localStorage.getItem('userInfo');
  
  if (authToken && savedUser) {
    const userData = JSON.parse(savedUser);
    globalUser.id = userData.userId;
    globalUser.token = authToken;
    globalUser.type = userData.type || 0; 
    globalUser.username = userData.username;
    globalUser.avatar_url = userData.avatar_url;
    globalUser.post_count = userData.postCount || 0;
    globalUser.following_count = userData.followCount || 0;
    globalUser.follower_count = userData.followerCount || 0;
    globalUser.type = userData.type || 0; // 默认普通用户类型
    globalUser.school_info = userData.school_info || ""; // 学校信息
  } else {
    globalUser.id = null;
    globalUser.username = "";
  }
  
  isLoggedIn = !!globalUser.id;
  updateProfileDisplay();
  updateNavigation(); // 新增：初始化导航栏显示
}

// 更新个人资料显示
function updateProfileDisplay() {
  const userProfileCard = document.getElementById('user-profile-card');
  const loginRegisterBtn = document.getElementById('login-register-btn');
  const userProfileNav = document.getElementById('user-profile-nav');
  
  if (globalUser.id) {
    // 更新隐藏用户相关元素
    const loginRequiredElements = document.querySelectorAll('.login-required');
    loginRequiredElements.forEach(element => {
        element.style.display = 'block';
        element.src = globalUser.avatar_url; 
    });

    // 已登录状态
    userProfileCard.classList.remove('hidden');
    loginRegisterBtn.classList.add('hidden');
    userProfileNav.classList.remove('hidden');
    
    // 更新导航栏用户信息
    document.getElementById('nav-avatar').src = globalUser.avatar_url;
    document.getElementById('nav-username').textContent = globalUser.username || '未设置';
    document.getElementById('nav-college').textContent = `${globalUser.school_info || '未设置'}`;
    document.getElementById('menu-username').textContent = globalUser.username || '个人主页';
    console.log('当前用户信息:', globalUser);
    // 更新侧边栏个人资料
    document.getElementById('sidebar-avatar').src = globalUser.avatar_url;
    document.getElementById('sidebar-username').textContent = globalUser.username || '未设置';
    document.getElementById('sidebar-college').textContent = `${globalUser.school_info || '未设置'}`;
    document.getElementById('post-count').textContent = globalUser.post_count;
    document.getElementById('follow-count').textContent = globalUser.following_count;
    document.getElementById('follower-count').textContent = globalUser.follower_count;
  } else {
    // 未登录状态
    userProfileCard.classList.add('hidden');
    loginRegisterBtn.classList.remove('hidden');
    userProfileNav.classList.add('hidden');
    const loginRequiredElements = document.querySelectorAll('.login-required');
    loginRequiredElements.forEach(element => {
        element.style.display = 'none';
    });
  }
}

// 更新导航栏显示
function updateNavigation() {
    const loginBtn = document.getElementById('login-register-btn');
    const userNav = document.getElementById('user-profile-nav');

    if (globalUser.id) {
        // 已登录：隐藏登录按钮，显示用户信息
        loginBtn.classList.add('hidden');
        userNav.classList.remove('hidden');
    } else {
        // 未登录：显示登录按钮，隐藏用户信息
        loginBtn.classList.remove('hidden');
        userNav.classList.add('hidden');
    }
}

// ------------------------------当前用户未登录相应对策-------------------
    function checkLoginStatusAndPrompt() {
        if (!globalUser.id) {
            alert('请先登录/注册以使用此功能');
            // 显示登录模态框（可选：自动弹出）
            document.getElementById('login-register-modal').classList.remove('hidden');
            document.body.style.overflow = 'hidden';
            return true; // 阻止后续操作
        }
        return false;
    }