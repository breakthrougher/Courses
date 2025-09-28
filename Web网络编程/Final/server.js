const express = require('express');  // 引入Express 框架
const mysql = require('mysql2/promise');      // 引入MySQL2的promise版本
const bodyParser = require('body-parser');
const cors = require('cors');
const fs = require('fs');
const dotenv = require('dotenv');
const bcrypt = require('bcryptjs'); // 用于密码加密
dotenv.config();

const app = express();
const PORT = 3000; 
app.use(cors());
app.use(bodyParser.json());

const multer = require('multer');
const path = require('path');

// 确保上传目录存在
const uploadDir = path.join(__dirname, 'public', 'uploads');
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}

// 配置Multer存储引擎
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, uploadDir);
    },
    filename: function (req, file, cb) {
        const uniqueSuffix = Date.now() + '_' + Math.round(Math.random() * 1e9);
        const ext = path.extname(file.originalname);
        cb(null, uniqueSuffix + ext);
    }
});
// 创建multer实例
const upload = multer({ 
    storage: storage,
    limits: { fileSize: 10 * 1024 * 1024 } // 限制文件大小为10MB
});

// 静态资源目录配置
app.use('/uploads', express.static(path.join(__dirname, 'public', 'uploads')));

let connection; // 声明连接变量

// 初始化数据库连接
async function initDb() {
    try {
        connection = await mysql.createConnection({
            host: 'localhost',
            user: 'root',
            password: '123456',
            database: 'web'
        });
        console.log('数据库连接成功');
    } catch (err) {
        console.error('数据库连接失败:', err);
        process.exit(1); // 连接失败时退出应用
    }
}

// 展示该数据库中所有表和视图，使用HTML，并给展示出来的所有表添加一个链接 点击进去可以查看该表的所有数据
// 展示数据库中所有表和视图，并添加链接
app.get('/', async (req, res) => {
    try {
        // 查询数据库中所有表
        const [tables] = await connection.execute('SHOW TABLES');
        
        let html = '<h1>数据库中的表:</h1><ul>';
        
        // 生成每个表的链接
        tables.forEach(table => {
            const tableName = table[`Tables_in_${process.env.DB_NAME || 'web'}`];
            html += `<li><a href="/table/${tableName}">${tableName}</a></li>`;
        });
        
        html += '</ul>';
        res.send(html);
    } catch (error) {
        console.error('查询表列表失败:', error);
        res.status(500).send('服务器内部错误');
    }
});

// 查看单个表的所有数据
app.get('/table/:tableName', async (req, res) => {
    try {
        const tableName = req.params.tableName;
        
        // 安全检查：确保表名只包含字母、数字和下划线
        if (!/^[a-zA-Z0-9_]+$/.test(tableName)) {
            return res.status(400).send('非法表名');
        }
        
        // 查询表结构
        const [columns] = await connection.execute(`DESCRIBE ${tableName}`);
        
        // 查询表数据
        const [rows] = await connection.execute(`SELECT * FROM ${tableName}`);
        
        // 生成HTML表格
        let html = `<h1>${tableName} 表数据</h1>`;
        html += '<a href="/">返回表列表</a><br><br>';
        
        if (rows.length === 0) {
            html += '<p>表中没有数据</p>';
        } else {
            html += '<table border="1" cellspacing="0" cellpadding="5">';
            
            // 表头
            html += '<tr>';
            columns.forEach(column => {
                html += `<th>${column.Field}</th>`;
            });
            html += '</tr>';
            
            // 表数据
            rows.forEach(row => {
                html += '<tr>';
                columns.forEach(column => {
                    html += `<td>${row[column.Field] || ''}</td>`;
                });
                html += '</tr>';
            });
            
            html += '</table>';
        }
        
        res.send(html);
    } catch (error) {
        console.error('查询表数据失败:', error);
        res.status(500).send('服务器内部错误');
    }
});

// 返回帖子和相关用户信息
app.get('/api/posts', async (req, res) => {
    try{
        const page = parseInt(req.query.page) || 1;
        const limit = parseInt(req.query.limit) || 10; 
        const offset = (page - 1) * limit;
        const currentUserId = req.query.userId || null; // 获取当前用户ID

        const query = `
            SELECT 
                p.post_id, p.media_url, p.content, p.created_at, p.like_count, p.comment_count, p.collect_count, p.updated_at,
                u.user_id, u.username, u.avatar_url, u.school_info,
                CASE WHEN l.user_id IS NOT NULL THEN 1 ELSE 0 END AS is_liked,
                CASE WHEN b.user_id IS NOT NULL THEN 1 ELSE 0 END AS is_collected,
                CASE WHEN f.follower_user_id IS NOT NULL THEN 1 ELSE 0 END AS is_following
            FROM posts p
            JOIN users u ON p.user_id = u.user_id
            LEFT JOIN Likes l ON p.post_id = l.post_id AND l.user_id = ?
            LEFT JOIN bookmarks b ON p.post_id = b.post_id AND b.user_id = ?
            LEFT JOIN Follows f ON u.user_id = f.following_user_id AND f.follower_user_id = ?
            ORDER BY p.created_at DESC
            LIMIT ? OFFSET ?;
        `;
        const [posts] = await connection.execute(query, [currentUserId+'', currentUserId+'',currentUserId+'', limit+'', offset+'']);
        console.log('查询帖子成功:', posts);

        const [totalResult] = await connection.execute('SELECT COUNT(*) as total FROM posts');
        const total = totalResult[0].total;
        res.json({
            data: posts,
            pagination: {
                currentPage: page,
                pageSize: limit,
                totalPages: Math.ceil(total / limit),
                totalRecords: total,
                hasMore: page < Math.ceil(total / limit)
            }
        });
    } catch (error) {
        console.error('查询帖子失败:', error);
        res.status(500).send('服务器内部错误');
    }
});

// 获取关注的用户的帖子
app.get('/api/followed_posts', async (req, res) => {
    try{
        const page = parseInt(req.query.page) || 1;
        const limit = parseInt(req.query.limit) || 10; 
        const offset = (page - 1) * limit;
        const currentUserId = req.query.userId || null; // 获取当前用户ID

        const query = `
            SELECT 
                p.post_id, p.media_url, p.content, p.created_at, p.like_count, p.comment_count, p.collect_count, p.updated_at,
                u.user_id, u.username, u.avatar_url, u.school_info,
                CASE WHEN l.user_id IS NOT NULL THEN 1 ELSE 0 END AS is_liked,
                CASE WHEN b.user_id IS NOT NULL THEN 1 ELSE 0 END AS is_collected,
                1 AS is_following
            FROM posts p
            JOIN users u ON p.user_id = u.user_id
            LEFT JOIN Likes l ON p.post_id = l.post_id AND l.user_id = ?
            LEFT JOIN bookmarks b ON p.post_id = b.post_id AND b.user_id = ?
            WHERE u.user_id IN (
                SELECT following_user_id 
                FROM Follows 
                WHERE follower_user_id = ?
            )
            ORDER BY p.created_at DESC
            LIMIT ? OFFSET ?;
        `;
        const [posts] = await connection.execute(query, [currentUserId+'', currentUserId+'',currentUserId+'',limit+'', offset+'']);
        console.log('查询帖子成功:', posts);

        // 查询总记录数
        const [totalResult] = await connection.execute(`
            SELECT COUNT(*) as total 
            FROM posts p
            JOIN users u ON p.user_id = u.user_id
            WHERE u.user_id IN (
                SELECT following_user_id 
                FROM Follows 
                WHERE follower_user_id = ?
            )
        `, [currentUserId]);
        const total = totalResult[0].total;
        res.json({
            data: posts,
            pagination: {
                currentPage: page,
                pageSize: limit,
                totalPages: Math.ceil(total / limit),
                totalRecords: total,
                hasMore: page < Math.ceil(total / limit)
            }
        });
    } catch (error) {
        console.error('查询帖子失败:', error);
        res.status(500).send('服务器内部错误');
    }
});

// 获取单个用户信息
app.get('/api/users/:userId', async (req, res) => {
    const userId = req.params.userId;
    try {
        const [user] = await connection.execute(
            'SELECT * FROM users WHERE user_id = ?',
            [userId]
        );
        if (user.length === 0) {
            return res.status(404).send('无效的用户ID');
        }
        res.json(user[0]);
    } catch (error) {
        console.error('查询用户失败:', error);
        res.status(500).send('服务器内部错误');
    }
});

// 图片上传接口
app.post('/api/upload_images', upload.array('images'), async (req, res) => {
    try {
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: '没有上传图片' });
        }
        console.log('接收到的文件数量:', req.files.length);
        console.log('文件详情:', req.files);
        const baseUrl = `${req.protocol}://${req.get('host')}`;
        const urls = req.files.map(file => {
            // 构建访问URL（public目录是静态资源根目录）
            return `${baseUrl}/uploads/${file.filename}`;
        });
        
        console.log('上传的图片路径:', urls);
        res.status(200).json({ urls });
    } catch (error) {
        console.error('图片上传失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 发布新帖子
app.post('/api/create_posts', upload.none(), async (req, res) => {
    try {
        // 从请求体获取数据
        const { user_id, content, media_url, location_info } = req.body;
        console.log('发布帖子数据:', req.body);
        // 数据验证
        if (!user_id || !content) {
            return res.status(400).json({ error: '用户ID和内容是必需的' });
        }
        
        let mediaUrls = media_url;
        if (typeof media_url === 'string') {
            try {
                mediaUrls = JSON.parse(media_url);
            } catch (e) {
                // 如果解析失败，当作单个URL处理
                mediaUrls = [media_url];
            }
        }
        
        // 将mediaUrls转换为JSON字符串存储
        const mediaUrlJson = JSON.stringify(mediaUrls || []);

        // 调用存储过程
        const [results] = await connection.execute(
            'CALL PublishPost(?, ?, ?, ?, @post_id)',
            [user_id, content, mediaUrlJson, location_info]
        );

        // 获取输出参数 @post_id是会话变量 需要执行一次查询才能获取
        const [output] = await connection.execute('SELECT @post_id AS post_id');
        const post_id = output[0].post_id;
        
        // 返回成功响应
        res.status(201).json({
            message: '帖子发布成功',
            post_id,
            user_id,
            content,
            mediaUrls,
            location_info,
            created_at: new Date().toISOString()
        });
    } catch (error) {
        console.error('发布帖子失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 删除帖子API
app.delete('/api/delete_posts/:postId', async (req, res) => {
    const postId = parseInt(req.params.postId);
    const { userId } = req.body; // 从请求体获取当前用户ID
    console.log('删除帖子请求:', { postId, userId });
    try {
        // 开启数据库事务
        await connection.beginTransaction();
        
        // 检查帖子是否存在
        const [post] = await connection.execute(
            'SELECT user_id FROM posts WHERE post_id = ?',
            [postId]
        );
        
        if (post.length === 0) {
            await connection.rollback();
            return res.status(404).json({ error: '帖子不存在' });
        }
        
        // 删除帖子相关的所有评论
        await connection.execute(
            'DELETE FROM comments WHERE post_id = ?',
            [postId]
        );

        // 删除帖子相关的所有点赞
        await connection.execute(
            'DELETE FROM Likes WHERE post_id = ?',
            [postId]
        );
        
        // 删除帖子相关的所有评论点赞
        await connection.execute(
            'DELETE FROM CommentLikes WHERE comment_id IN (SELECT comment_id FROM comments WHERE post_id = ?)',
            [postId]
        );

        // 删除帖子相关的所有收藏
        await connection.execute(
            'DELETE FROM bookmarks WHERE post_id = ?',
            [postId]
        );

        // 最后删除帖子本身
        await connection.execute(
            'DELETE FROM posts WHERE post_id = ?',
            [postId]
        );
        
        // 用户相关的帖子数减一
        await connection.execute(
            'UPDATE users SET post_count = post_count - 1 WHERE user_id = ?',
            [userId]
        );

        // 提交事务
        await connection.commit();
        
        res.status(200).json({ 
            message: '帖子删除成功',
            postId
        });
    } catch (error) {
        // 发生错误时回滚事务
        await connection.rollback();
        console.error('删除帖子失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 点赞帖子
app.post('/api/like/:postId',async (req, res) => {
    const postId = req.params.postId;
    const { user_id } = req.body; // 从请求体获取用户ID

    try {
        // 检查是否已点赞（避免重复点赞）
        const [exists] = await connection.execute(
            'SELECT * FROM `Likes` WHERE `post_id` = ? AND `user_id` = ?',
            [postId, user_id]
        );
        
        if (exists.length > 0) {
            return res.status(400).json({ error: '已点赞过该帖子' });
        }
        
        // 开启事务
        await connection.beginTransaction();
        
        // 添加点赞记录
        await connection.execute(
            'INSERT INTO `Likes` (`post_id`, `user_id`) VALUES (?, ?)',
            [postId, user_id]
        );
        console.log("success");
        // 更新帖子点赞数
        await connection.execute(
            'UPDATE `Posts` SET `like_count` = `like_count` + 1 WHERE `post_id` = ?',
            [postId]
        );

        const [result] = await connection.execute('SELECT COUNT(*) as like_count FROM `Likes` WHERE `post_id` = ?', [postId]);

        await connection.commit();
        
        res.json({
            likeCount: result[0].like_count
        });
    } catch (error) {
        await connection.rollback();
        console.error('点赞失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 取消点赞帖子
app.delete('/api/like/:postId',async (req, res) => {
    const postId = req.params.postId;
    const { user_id } = req.body; // 从请求体获取用户ID
    
    try {
        await connection.beginTransaction();
        
        // 删除点赞记录
        await connection.execute(
            'DELETE FROM `Likes` WHERE `post_id` = ? AND `user_id` = ?',
            [postId, user_id]
        );
        console.log("fail");
        // 更新帖子点赞数
        await connection.execute(
            'UPDATE `Posts` SET `like_count` = `like_count` - 1 WHERE `post_id` = ?',
            [postId]
        );
        const [result] = await connection.execute('SELECT COUNT(*) as like_count FROM `Likes` WHERE `post_id` = ?', [postId]);
        await connection.commit();
        
        res.json({
            likeCount: result[0].like_count
        });
    } catch (error) {
        await connection.rollback();
        console.error('取消点赞失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 获取评论
app.get('/api/comments/:postId', async (req, res) => {
    const postId = parseInt(req.params.postId);
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 5;
    const offset = (page - 1) * limit;
    const currentUserId = req.query.userId || null; // 获取当前用户ID

    try {
        // 查询评论及用户信息
        const [comments] = await connection.execute(`
            SELECT 
                c.comment_id, c.post_id, c.user_id, c.content, c.updated_at, c.like_count,
                u.user_id, u.username, u.avatar_url,
                CASE WHEN cl.user_id IS NOT NULL THEN 1 ELSE 0 END AS is_liked
            FROM comments c
            JOIN users u ON c.user_id = u.user_id
            LEFT JOIN CommentLikes cl ON c.comment_id = cl.comment_id AND cl.user_id = ?
            WHERE c.post_id = ?
            ORDER BY c.updated_at DESC
            LIMIT ? OFFSET ?
        `, [currentUserId+'', postId+'', limit+'', offset+'']);
        
        const [totalResult] = await connection.execute(
            'SELECT COUNT(*) as total FROM comments WHERE post_id = ?',
            [postId]
        );
        console.log('查询评论成功:', comments);
        res.json({
            data: comments,
            pagination: {
                currentPage: page,
                pageSize: limit,
                totalPages: Math.ceil(totalResult[0].total / limit),
                totalRecords: totalResult[0].total,
                hasMore: page < Math.ceil(totalResult[0].total / limit)
            }
        });
    } catch (error) {
        console.error('查询评论失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 发布评论
app.post('/api/create_comments/:postId', async (req, res) => {
    const postId = parseInt(req.params.postId);
    const { user_id, content } = req.body;

    if (!user_id || !content) {
        return res.status(400).json({ error: '用户ID和评论内容是必需的' });
    }

    try {
        await connection.beginTransaction();
        
        // 添加评论
        const [result] = await connection.execute(
            'INSERT INTO comments (post_id, user_id, content) VALUES (?, ?, ?)',
            [postId, user_id, content]
        );
        
        // 更新帖子评论数
        await connection.execute(
            'UPDATE posts SET comment_count = comment_count + 1 WHERE post_id = ?',
            [postId]
        );
        
        // 查询新添加的评论
        const [newComment] = await connection.execute(`
            SELECT 
                c.comment_id, c.post_id, c.user_id, c.content, c.created_at,
                u.username, u.avatar_url
            FROM comments c
            JOIN users u ON c.user_id = u.user_id
            WHERE c.comment_id = ?
        `, [result.insertId]);
        
        await connection.commit();
        
        res.status(201).json({
            message: '评论添加成功',
            comment: newComment[0] // 返回新添加的评论
        });
    } catch (error) {
        await connection.rollback();
        console.error('添加评论失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 删除评论
app.delete('/api/delete_comments/:commentId', async (req, res) => {
    const commentId = parseInt(req.params.commentId);
    
    try {
        // 开启数据库事务
        await connection.beginTransaction();
        
        // 1. 检查评论是否存在
        const [commentResult] = await connection.execute(
            'SELECT post_id FROM comments WHERE comment_id = ?',
            [commentId]
        );
        
        if (commentResult.length === 0) {
            await connection.rollback();
            return res.status(404).json({ error: '评论不存在' });
        }
        
        const comment = commentResult[0];
        const commentPostId = comment.post_id;
        
        // 2. 删除评论相关的所有点赞
        await connection.execute(
            'DELETE FROM CommentLikes WHERE comment_id = ?',
            [commentId]
        );
        
        // 3. 删除评论本身
        await connection.execute(
            'DELETE FROM comments WHERE comment_id = ?',
            [commentId]
        );
        
        // 4. 更新帖子的评论数
        await connection.execute(
            'UPDATE posts SET comment_count = comment_count - 1 WHERE post_id = ?',
            [commentPostId]
        );
        
        // 提交事务
        await connection.commit();
        
        res.status(200).json({ 
            message: '评论删除成功',
            commentId
        });
    } catch (error) {
        // 发生错误时回滚事务
        await connection.rollback();
        console.error('删除评论失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 点赞评论
app.post('/api/comment/like/:commentId', async (req, res) => {
    const commentId = req.params.commentId;
    const { user_id } = req.body; // 从请求体获取用户ID

    try {
        // 检查是否已点赞（避免重复点赞）
        const [exists] = await connection.execute(
            'SELECT * FROM `CommentLikes` WHERE `comment_id` = ? AND `user_id` = ?',
            [commentId, user_id]
        );
        
        if (exists.length > 0) {
            return res.status(400).json({ error: '已点赞过该评论' });
        }
        
        // 开启事务
        await connection.beginTransaction();
        
        // 添加点赞记录
        await connection.execute(
            'INSERT INTO `CommentLikes` (`comment_id`, `user_id`) VALUES (?, ?)',
            [commentId, user_id]
        );
        
        // 更新评论点赞数
        await connection.execute(
            'UPDATE `Comments` SET `like_count` = `like_count` + 1 WHERE `comment_id` = ?',
            [commentId]
        );

        const [result] = await connection.execute('SELECT COUNT(*) as like_count FROM `CommentLikes` WHERE `comment_id` = ?', [commentId]);

        await connection.commit();
        
        res.json({
            likeCount: result[0].like_count,
            isLiked: true
        });
    } catch (error) {
        await connection.rollback();
        console.error('点赞评论失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 取消评论点赞
app.delete('/api/comment/like/:commentId', async (req, res) => {
    const commentId = req.params.commentId;
    const { user_id } = req.body; // 从请求体获取用户ID
    
    try {
        await connection.beginTransaction();
        
        // 删除点赞记录
        await connection.execute(
            'DELETE FROM `CommentLikes` WHERE `comment_id` = ? AND `user_id` = ?',
            [commentId, user_id]
        );
        
        // 更新评论点赞数
        await connection.execute(
            'UPDATE `Comments` SET `like_count` = `like_count` - 1 WHERE `comment_id` = ?',
            [commentId]
        );
        
        const [result] = await connection.execute('SELECT COUNT(*) as like_count FROM `CommentLikes` WHERE `comment_id` = ?', [commentId]);
        
        await connection.commit();
        
        res.json({
            likeCount: result[0].like_count,
            isLiked: false
        });
    } catch (error) {
        await connection.rollback();
        console.error('取消点赞评论失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 收藏帖子
app.post('/api/collect/:postId', async (req, res) => {
    const postId = parseInt(req.params.postId);
    const { user_id } = req.body; // 从请求体获取用户ID

    try {
        // 数据验证
        if (!user_id) {
            return res.status(400).json({ error: '用户ID是必需的' });
        }

        // 开启事务
        await connection.beginTransaction();
        
        // 检查是否已收藏（避免重复收藏）
        const [exists] = await connection.execute(
            'SELECT * FROM `bookmarks` WHERE `post_id` = ? AND `user_id` = ?',
            [postId, user_id]
        );

        if (exists.length > 0) {
            await connection.rollback();
            return res.status(400).json({ error: '已收藏过该帖子' });
        }

        // 添加收藏记录
        await connection.execute(
            'INSERT INTO `bookmarks` (`post_id`, `user_id`) VALUES (?, ?)',
            [postId, user_id]
        );

        // 更新帖子收藏数
        await connection.execute(
            'UPDATE `Posts` SET `collect_count` = `collect_count` + 1 WHERE `post_id` = ?',
            [postId]
        );

        // 获取更新后的收藏数
        const [result] = await connection.execute(
            'SELECT `collect_count` FROM `Posts` WHERE `post_id` = ?',
            [postId]
        );

        await connection.commit();

        res.json({
            message: '收藏成功',
            collectCount: result[0].collect_count,
            isCollected: true
        });
    } catch (error) {
        await connection.rollback();
        console.error('收藏失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 取消收藏帖子
app.delete('/api/collect/:postId', async (req, res) => {
    const postId = parseInt(req.params.postId);
    const { user_id } = req.body; // 从请求体获取用户ID
    
    try {
        // 数据验证
        if (!user_id) {
            return res.status(400).json({ error: '用户ID是必需的' });
        }

        await connection.beginTransaction();
        
        // 删除收藏记录
        await connection.execute(
            'DELETE FROM `bookmarks` WHERE `post_id` = ? AND `user_id` = ?',
            [postId, user_id]
        );
        
        // 更新帖子收藏数
        await connection.execute(
            'UPDATE `Posts` SET `collect_count` = `collect_count` - 1 WHERE `post_id` = ?',
            [postId]
        );
        
        // 获取更新后的收藏数
        const [result] = await connection.execute(
            'SELECT `collect_count` FROM `Posts` WHERE `post_id` = ?',
            [postId]
        );
        
        await connection.commit();
        
        res.json({
            message: '取消收藏成功',
            collectCount: result[0].collect_count,
            isCollected: false
        });
    } catch (error) {
        await connection.rollback();
        console.error('取消收藏失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 关注用户
app.post('/api/follow/:userId', async (req, res) => {
    const { followerId } = req.body;
    const followingId = parseInt(req.params.userId); // 被关注用户ID
    console.log('关注请求:', { followerId, followingId });
    try {
        // 数据验证
        if (!followerId || !followingId) {
            return res.status(400).json({ error: '用户ID是必需的' });
        }
        
        // 检查是否已关注
        const [exists] = await connection.execute(
            'SELECT * FROM Follows WHERE follower_user_id = ? AND following_user_id = ?',
            [followerId, followingId]
        );
        
        if (exists.length > 0) {
            return res.status(400).json({ error: '已关注该用户' });
        }
        
        // 开启事务
        await connection.beginTransaction();
        
        // 添加关注记录
        await connection.execute(
            'INSERT INTO Follows (follower_user_id, following_user_id) VALUES (?, ?)',
            [followerId, followingId]
        );
        
        // 更新被关注者的粉丝数
        await connection.execute(
            'UPDATE Users SET follower_count = follower_count + 1 WHERE user_id = ?',
            [followingId]
        );
        
        // 更新关注者的关注数
        await connection.execute(
            'UPDATE Users SET following_count = following_count + 1 WHERE user_id = ?',
            [followerId]
        );
        
        // 获取更新后的关注状态
        const [result] = await connection.execute(
            'SELECT * FROM Follows WHERE follower_user_id = ? AND following_user_id = ?',
            [followerId, followingId]
        );
        
        await connection.commit();
        
        res.json({
            message: '关注成功',
            followId: result.insertId,
            isFollowing: true
        });
    } catch (error) {
        await connection.rollback();
        console.error('关注失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 取消关注用户
app.delete('/api/follow/:userId', async (req, res) => {
    const { followerId } = req.body;
    const followingId = parseInt(req.params.userId); // 被关注用户ID
    
    try {
        // 数据验证
        if (!followerId || !followingId) {
            return res.status(400).json({ error: '用户ID是必需的' });
        }
        
        // 开启事务
        await connection.beginTransaction();
        
        // 检查是否已关注
        const [exists] = await connection.execute(
            'SELECT * FROM Follows WHERE follower_user_id = ? AND following_user_id = ?',
            [followerId, followingId]
        );
        
        if (exists.length === 0) {
            await connection.rollback();
            return res.status(400).json({ error: '未关注该用户' });
        }
        
        // 删除关注记录
        await connection.execute(
            'DELETE FROM Follows WHERE follower_user_id = ? AND following_user_id = ?',
            [followerId, followingId]
        );
        
        // 更新被关注者的粉丝数
        await connection.execute(
            'UPDATE Users SET follower_count = follower_count - 1 WHERE user_id = ?',
            [followingId]
        );
        
        // 更新关注者的关注数
        await connection.execute(
            'UPDATE Users SET following_count = following_count - 1 WHERE user_id = ?',
            [followerId]
        );
        
        await connection.commit();
        
        res.json({
            message: '取消关注成功',
            isFollowing: false
        });
    } catch (error) {
        await connection.rollback();
        console.error('取消关注失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 获取用户关注列表
app.get('/api/getFollows/:userId', async (req, res) => {
    try {
        const userId = req.params.userId;
        const [follows] = await connection.execute(`
            SELECT 
                f.following_user_id as followedId,
                u.user_id,
                u.username,
                u.avatar_url,
                u.school_info
            FROM Follows f
            JOIN users u ON f.following_user_id = u.user_id
            WHERE f.follower_user_id = ?
            ORDER BY f.created_at DESC
        `, [userId]);
        console.log('获取关注列表成功:', follows);

        res.json({
            data: follows,
            total: follows.length
        });
    } catch (error) {
        console.error('查询关注列表失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 获取点赞列表
app.get('/api/getLikes/:userId', async (req, res) => {
    try {
        const userId = req.params.userId;
        
        // 构建查询语句，获取用户点赞的帖子及相关信息
        const query = `
            SELECT 
                l.like_id, l.post_id, l.user_id as liker_id, l.created_at,
                u.user_id, u.username, u.avatar_url, u.school_info,
                p.post_id as post_id, p.content as post_content
            FROM Likes l
            JOIN Users u ON l.user_id = u.user_id  -- 点赞的用户
            JOIN Posts p ON l.post_id = p.post_id  -- 被点赞的帖子
            WHERE p.user_id = ?                    -- 帖子属于当前用户
            ORDER BY l.created_at DESC
        `;
        
        const [likes] = await connection.execute(query, [userId]);
        console.log('获取点赞列表成功:', likes);
           
        res.json({
            data: likes,
            total: likes.length,
            message: '获取点赞列表成功'
        });
    } catch (error) {
        console.error('查询点赞列表失败:', error);
        res.status(500).json({ 
            error: '服务器内部错误',
            message: '获取点赞列表失败，请稍后重试'
        });
    }
});

// 获取用户收藏的帖子
app.get('/api/getCollects/:userID', async (req, res) => {
    const userId = req.params.userID; // 从请求参数获取用户ID
    
    if (!userId) {
        return res.status(400).json({ error: '用户ID是必需的' });
    }

    try {
        const page = parseInt(req.query.page) || 1;
        const limit = parseInt(req.query.limit) || 10; 
        const offset = (page - 1) * limit;

        // 查询用户收藏的帖子
        const [collections] = await connection.execute(`
            SELECT
                p.post_id, p.media_url, p.content, p.created_at, p.like_count, p.comment_count, p.collect_count, p.updated_at,
                u.user_id, u.username, u.avatar_url, u.school_info,
                1 AS is_collected
            FROM bookmarks c
            JOIN Posts p ON c.post_id = p.post_id
            JOIN Users u ON p.user_id = u.user_id
            WHERE c.user_id = ?
            ORDER BY p.created_at DESC
            LIMIT ? OFFSET ?;
        `, [userId+'', limit+'', offset+'']);

        const [totalResult] = await connection.execute(`
            SELECT COUNT(*) as total
            FROM bookmarks c
            WHERE c.user_id = ?
        `, [userId]);
        const total = totalResult[0].total;

        res.json({
            data: collections,
            pagination: {
                currentPage: page,
                pageSize: limit,
                totalPages: Math.ceil(total / limit),
                totalRecords: total,
                hasMore: page < Math.ceil(total / limit)
            }
        });
    } catch (error) {
        console.error('查询收藏失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 推荐用户关注API
app.get('/api/recommend_users/:userId', async (req, res) => {
    try {
        const userId = req.params.userId;
        const page = parseInt(req.query.page) || 1;
        const limit = parseInt(req.query.limit) || 3; // 最多返回3个推荐用户
        const offset = (page - 1) * limit;
        
        //    - 排除已关注用户和自己
        //    - 可以基于共同关注、相似兴趣等条件推荐
        //    - 这里使用简单示例：随机选择未关注用户
        const query = `
            SELECT 
                u.user_id, u.username, u.avatar_url, u.school_info,
                u.follower_count, u.post_count,
                0 as is_following
            FROM users u
            WHERE u.user_id NOT IN (SELECT following_user_id FROM Follows WHERE follower_user_id = ?)
                and u.user_id != ?
            ORDER BY 
                u.follower_count DESC, 
                u.post_count DESC,
                RAND() -- 随机排序，增加多样性
            LIMIT ? OFFSET ?
        `;
        
        const [users] = await connection.execute(query, [userId+'', userId+'', limit+'', offset+'']);
        
        // 3. 查询总推荐用户数（排除已关注用户和自己）
        const countQuery = `
            SELECT COUNT(*) as total
            FROM users u
            WHERE u.user_id NOT IN (SELECT following_user_id FROM Follows WHERE follower_user_id = ?)
                and u.user_id != ?
        `;
        const countParams = [userId+'', userId+''];
        const [countResult] = await connection.execute(countQuery, countParams);
        const total = countResult[0].total;
        
        console.log('查询推荐用户成功:', users);
        res.json({
            data: users,
            pagination: {
                currentPage: page,
                pageSize: limit,
                totalPages: Math.ceil(total / limit),
                totalRecords: total,
                hasMore: page < Math.ceil(total / limit)
            },
            message: '获取推荐用户成功'
        });
    } catch (error) {
        console.error('查询推荐用户失败:', error);
        res.status(500).json({ 
            error: '服务器内部错误',
            message: '获取推荐用户失败，请稍后重试'
        });
    }
});

// 用户注册API
app.post('/api/register', upload.none(), async (req, res) => {
    try {
        // 验证请求数据
        const { name, email, bio, password, avatarUrl } = req.body;
        
        if (!name || !email || !password) {
            return res.status(400).json({ error: '缺少必要字段' });
        }
        
        // 检查邮箱格式（简单验证）
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(email)) {
            return res.status(400).json({ error: '邮箱格式不正确' });
        }
        
        // 检查邮箱是否已被注册
        const [users] = await connection.execute(
            'SELECT user_id FROM users WHERE email = ?',
            [email]
        );
        
        if (users.length > 0) {
            return res.status(400).json({ error: '邮箱已被注册' });
        }
        
        // 生成密码哈希
        const saltRounds = 10;
        const hashedPassword = await bcrypt.hash(password, saltRounds);
        
        // 插入用户数据到数据库
        const [result] = await connection.execute(
            'INSERT INTO users (username, email, bio, password_hash, avatar_url) VALUES (?, ?, ?, ?, ?)',
            [name, email, bio, hashedPassword, avatarUrl]
        );
        
        // 获取插入的用户ID
        const userId = result.insertId;
        
        // 返回成功响应
        res.status(201).json({
            message: '注册成功',
            user: {
                id: userId,
                name,
                email,
                bio,
                avatarUrl,
                created_at: new Date().toISOString()
            }
        });
    } catch (error) {
        console.error('注册失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 用户登录API
app.post('/api/login', async (req, res) => {
    try {
        const { email, password } = req.body;
        
        if (!email || !password) {
            return res.status(400).json({ error: '缺少必要字段' });
        }
        
        // 查询用户
        const [users] = await connection.execute(
            'SELECT user_id, username, avatar_url, password_hash, type, post_count, follower_count, following_count,school_info FROM users WHERE email = ?',
            [email]
        );
        
        if (users.length === 0) {
            return res.status(401).json({ error: '邮箱或密码错误' });
        }
        
        const user = users[0];
        
        // 验证密码
        const isPasswordValid = await bcrypt.compare(password, user.password_hash);
        
        if (!isPasswordValid) {
            return res.status(401).json({ error: '邮箱或密码错误' });
        }
        
        delete user.password_hash;

        res.status(200).json({
            message: '登录成功',
            user
        });
    } catch (error) {
        console.error('登录失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 返回帖子和相关用户信息
app.get('/api/posts/simple', async (req, res) => {
    try {
        // 1. 将输入转换为安全的整数，这是防止SQL注入的关键
        const page = parseInt(req.query.page) || 1;
        const limit = parseInt(req.query.limit) || 10;
        const offset = (page - 1) * limit;
        // 2. 因为LIMIT/OFFSET不支持 '?' 占位符，我们直接将安全的整数拼接到SQL字符串中
        const sql = `
            SELECT p.post_id, p.user_id, p.media_url, p.content, p.created_at, p.like_count, p.comment_count, u.username, u.avatar_url, u.school_info, u.gender
            FROM posts p
            JOIN users u ON p.user_id = u.user_id
            ORDER BY p.created_at DESC
            LIMIT ${limit} OFFSET ${offset}
        `;
        // 3. 执行SQL时，不再需要传递参数数组
        const [posts] = await connection.execute(sql);

        const [totalResult] = await connection.execute('SELECT COUNT(*) as total FROM posts');
        const total = totalResult[0].total;

        res.json({
            data: posts,
            pagination: {
                currentPage: page,
                pageSize: limit,
                totalPages: Math.ceil(total / limit),
                totalRecords: total,
                hasMore: page < Math.ceil(total / limit)
            }
        });
    } catch (error) {
        console.error('查询帖子失败:', error);
        res.status(500).json({ error: '服务器内部错误', details: error.message });
    }
});
// [替换为这个新的] 更新单个帖子的接口，现在支持图片上传
app.put('/api/post/update', upload.array('newImages'), async (req, res) => {
    try {
        // 1. 从请求体和文件中获取数据
        const { postId, content, loggedInUserId, existingImages } = req.body;
        const newImageFiles = req.files; // 这是 multer 处理后的文件数组

        // 2. 数据验证
        if (!postId || !loggedInUserId) {
            return res.status(400).json({ error: '缺少必要参数 (postId, loggedInUserId)' });
        }

        // --- 核心安全验证 (和之前一样) ---
        const [posts] = await connection.execute('SELECT user_id, media_url FROM Posts WHERE post_id = ?', [postId]);
        if (posts.length === 0) return res.status(404).json({ error: '帖子不存在' });
        
        const authorId = posts[0].user_id;
        if (authorId !== parseInt(loggedInUserId)) {
            return res.status(403).json({ error: '权限不足！' });
        }
        // --- 安全验证结束 ---
        const baseUrl = `${req.protocol}://${req.get('host')}`;
        // 3. 处理图片URL
        let finalImageUrls = [];
        // a. 解析要保留的旧图片URL
        if (existingImages) {
            try {
                const parsedExisting = JSON.parse(existingImages);
                finalImageUrls.push(...parsedExisting);
            } catch(e) { /* 忽略解析错误 */ }
        }
        
        // 处理新上传的图片
        if (newImageFiles && newImageFiles.length > 0) {
            const newUrls = newImageFiles.map(file => `${baseUrl}/uploads/${file.filename}`);
            finalImageUrls.push(...newUrls);
        }

        // 4. 执行数据库更新
        // 将最终的URL数组转为JSON字符串存入数据库
        const mediaUrlJson = JSON.stringify(finalImageUrls);

        await connection.execute(
            'UPDATE Posts SET content = ?, media_url = ?, updated_at = NOW() WHERE post_id = ?',
            [content, mediaUrlJson, postId]
        );

        // (可选但推荐) 删除不再使用的旧图片文件
        // const oldImageUrls = JSON.parse(posts[0].media_url || '[]');
        // oldImageUrls.forEach(oldUrl => {
        //     if (!finalImageUrls.includes(oldUrl)) {
        //         const imagePath = path.join(__dirname, 'public', oldUrl);
        //         fs.unlink(imagePath, (err) => {
        //             if (err) console.error(`删除旧图片失败: ${imagePath}`, err);
        //         });
        //     }
        // });

        res.status(200).json({ message: '帖子更新成功', updatedUrls: finalImageUrls });

    } catch (error) {
        console.error('更新帖子失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});
// [新增] 删除单个帖子的接口
app.delete('/api/post/delete', async (req, res) => {
    try {
        console.log('删除帖子请求:', req.body);
        // 1. 从请求体获取数据
        const { postId, loggedInUserId } = req.body;

        // 2. 数据验证
        if (!postId || !loggedInUserId) {
            return res.status(400).json({ error: '缺少必要参数 (postId, loggedInUserId)' });
        }

        if (posts.length === 0) {
            // 即使帖子不存在，也返回成功，避免信息泄露，或者直接返回404
            return res.status(404).json({ error: '帖子不存在' });
        }
        
        // 删除帖子相关的所有评论
        await connection.execute(
            'DELETE FROM comments WHERE post_id = ?',
            [postId]
        );

        // 删除帖子相关的所有点赞
        await connection.execute(
            'DELETE FROM Likes WHERE post_id = ?',
            [postId]
        );
        
        // 删除帖子相关的所有评论点赞
        await connection.execute(
            'DELETE FROM CommentLikes WHERE comment_id IN (SELECT comment_id FROM comments WHERE post_id = ?)',
            [postId]
        );

        // 删除帖子相关的所有收藏
        await connection.execute(
            'DELETE FROM bookmarks WHERE post_id = ?',
            [postId]
        );

        // 最后删除帖子本身
        await connection.execute(
            'DELETE FROM posts WHERE post_id = ?',
            [postId]
        );
        
        // 用户相关的帖子数减一
        await connection.execute(
            'UPDATE users SET post_count = post_count - 1 WHERE user_id = ?',
            [userId]
        );

        if (result.affectedRows > 0) {
            res.status(200).json({ message: '帖子删除成功' });
        } else {
            // 这通常在并发操作时发生，帖子已经被别人删了
            res.status(404).json({ error: '帖子不存在或已被删除' });
        }

    } catch (error) {
        console.error('删除帖子失败:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});
// [新增] 更新用户资料的接口
app.put('/api/user/update', upload.none(), async (req, res) => {
    // upload.none() 用于处理 multipart/form-data 但不包含文件上传
    console.log('更新用户资料请求:', req.body);
    try {
        // 1. 从请求体中获取所有可能被更新的字段
        const {
            userId,
            username,
            email,
            school_info,
            bio,
            phone_number,
            enrollment_date,
            gender
        } = req.body;

        // 2. 数据验证：确保必要的userId存在
        if (!userId) {
            return res.status(400).json({ error: '用户ID是必需的' });
        }

        // --- 安全提示 ---
        // 在真实的应用中，这里必须验证当前操作者是否有权限修改 userId 对应的用户。
        // 例如：const loggedInUserId = req.session.userId;
        // if (loggedInUserId !== parseInt(userId)) {
        //     return res.status(403).json({ error: '无权修改此用户' });
        // }

        // 3. 动态构建 SQL UPDATE 语句
        const fieldsToUpdate = [];
        const values = [];

        // 检查每个字段是否存在，如果存在，则添加到更新列表中
        if (username) { fieldsToUpdate.push('username = ?'); values.push(username); }
        if (email) { fieldsToUpdate.push('email = ?'); values.push(email); }
        if (school_info) { fieldsToUpdate.push('school_info = ?'); values.push(school_info); }
        if (bio) { fieldsToUpdate.push('bio = ?'); values.push(bio); }
        if (phone_number) { fieldsToUpdate.push('phone_number = ?'); values.push(phone_number); }
        if (enrollment_date) { fieldsToUpdate.push('enrollment_date = ?'); values.push(enrollment_date); }
        if (gender) { fieldsToUpdate.push('gender = ?'); values.push(gender); }

        // 如果没有任何字段需要更新，直接返回成功
        if (fieldsToUpdate.length === 0) {
            return res.json({ message: '没有需要更新的字段' });
        }

        // 将 userId 添加到值的末尾，用于 WHERE 子句
        values.push(userId);

        // 4. 拼接并执行 SQL
        const sql = `UPDATE Users SET ${fieldsToUpdate.join(', ')} WHERE user_id = ?`;

        const [result] = await connection.execute(sql, values);

        if (result.affectedRows > 0) {
            // [修改这里] 更新成功后，立刻查询最新的用户信息并返回
            const [updatedUsers] = await connection.execute(
                'SELECT user_id, username, gender, avatar_url, school_info, bio, interests_tags, phone_number, enrollment_date, post_count, follower_count, following_count, email FROM users WHERE user_id = ?',
                [userId]
            );
            res.json({
                message: '个人资料更新成功！',
                updatedUser: updatedUsers[0] // 将更新后的用户对象返回给前端
            });
        } else {
            res.status(404).json({ error: '未找到要更新的用户' });
        }

    } catch (error) {
        console.error('更新用户资料失败:', error);
        // 检查是否是唯一键冲突 (例如用户名或邮箱已存在)
        if (error.code === 'ER_DUP_ENTRY') {
            return res.status(409).json({ error: '用户名或邮箱已存在，请使用其他值。' });
        }
        res.status(500).json({ error: '服务器内部错误' });
    }
});
// 获取单个用户信息和其发布的帖子
app.get('/api/user/:userId', async (req, res) => {
    const userId = req.params.userId;
    try {
        // [查询1] 获取用户信息
        const [users] = await connection.execute(
            'SELECT user_id, username,email, avatar_url, gender, school_info, bio, interests_tags, phone_number, enrollment_date, post_count, follower_count, following_count FROM users WHERE user_id = ?',
            [userId]
        );

        if (users.length === 0) {
            return res.status(404).json({ error: '用户不存在' });
        }
        const userDetails = users[0];

        // [查询2] 获取该用户发布的所有帖子
        const [userPosts] = await connection.execute(`
            SELECT post_id, user_id, content, media_url, created_at, like_count, comment_count 
            FROM posts 
            WHERE user_id = ? 
            ORDER BY created_at DESC
        `, [userId]);

        // 组合数据并返回
        res.json({
            userDetails: userDetails,
            userPosts: userPosts
        });

    } catch (error) {
        console.error(`查询用户 ${userId} 的数据失败:`, error);
        res.status(500).send('服务器内部错误');
    }
});


app.post('/api/checkin', async (req, res) => {
  try {
    const { userId, comment, clientDate } = req.body; // 添加 clientDate 参数
    
    if (!userId) {
      return res.status(400).json({ error: '用户ID是必需的' });
    }
    
    const dateStr = clientDate || new Date().toISOString().split('T')[0];
    
    // 检查今天是否已经打卡
    const [existingCheckin] = await connection.execute(
      'SELECT * FROM UserCheckins WHERE user_id = ? AND checkin_date = ?',
      [userId, dateStr]
    );
    
    if (existingCheckin.length > 0) {
      return res.status(400).json({ 
        success: false,
        message: '今天已经打卡过了'
      });
    }
    
    // 添加打卡记录
    await connection.execute(
      'INSERT INTO UserCheckins (user_id, checkin_date, comment) VALUES (?, ?, ?)',
      [userId, dateStr, comment || null]
    );
    
    res.status(201).json({
      success: true,
      message: '打卡成功',
      date: dateStr
    });
  } catch (error) {
    console.error('打卡失败:', error);
    res.status(500).json({ 
      success: false,
      message: '服务器内部错误'
    });
  }
});
// 获取用户某月的打卡记录
app.get('/api/checkins', async (req, res) => {
  try {
    const { userId, year, month } = req.query;
    
    if (!userId || !year || !month) {
      return res.status(400).json({ error: '用户ID、年份和月份是必需的' });
    }
    
    const startDate = `${year}-${String(month).padStart(2, '0')}-01`;
    const lastDay = new Date(year, month, 0).getDate();
    const endDate = `${year}-${String(month).padStart(2, '0')}-${String(lastDay).padStart(2, '0')}`;
    
    // 修改SQL查询，使用DATE_FORMAT
    // 并确保返回的字段名为 checkin_date，或者调整客户端代码以匹配新字段名
    const [rawCheckins] = await connection.execute(
      `SELECT 
         user_id, 
         DATE_FORMAT(checkin_date, '%Y-%m-%d') as checkin_date,  -- 直接将格式化后的日期赋给 checkin_date
         comment,
         checkin_id -- 假设有这个主键或其他字段你可能需要
       FROM UserCheckins 
       WHERE user_id = ? AND checkin_date BETWEEN ? AND ? 
       ORDER BY checkin_date`,
      [userId, startDate, endDate]
    );
    
    const [streakResult] = await connection.execute(
      'SELECT COUNT(DISTINCT DATE(checkin_date)) as streak_count ' + // 这是一个简化的连续打卡近似值，实际连续打卡更复杂
      'FROM UserCheckins WHERE user_id = ? AND ' +
      'checkin_date >= (SELECT MAX(checkin_date) - INTERVAL 30 DAY FROM UserCheckins WHERE user_id = ?)',
      [userId, userId]
    );

    
    const streakCount = streakResult[0]?.streak_count || 0;
    
    res.json({
      success: true,
      data: rawCheckins, // 直接使用 rawCheckins，因为日期已被格式化
      streakCount: Math.max(0, streakCount), // 保持 streakCount 不变，但要注意其准确性
      totalDays: rawCheckins.length // totalDays 也基于查询结果
    });
  } catch (error) {
    console.error('获取打卡记录失败:', error);
    res.status(500).json({ error: '服务器内部错误' });
  }
});
// 同时，获取指定日期打卡记录的接口也应该做类似处理
app.get('/api/checkins/date', async (req, res) => {
  try {
    const { userId, date } = req.query;
    
    if (!userId || !date) {
      return res.status(400).json({ error: '用户ID和日期是必需的' });
    }
    
    const [checkins] = await connection.execute(
      `SELECT 
         user_id, 
         DATE_FORMAT(checkin_date, '%Y-%m-%d') as checkin_date, 
         comment,
         checkin_id
       FROM UserCheckins 
       WHERE user_id = ? AND DATE(checkin_date) = ?`, // 确保比较的是日期部分
      [userId, date]
    );
    
    res.json({
      success: true,
      data: checkins.length > 0 ? checkins[0] : null
    });
  } catch (error) {
    console.error('获取打卡记录失败:', error);
    res.status(500).json({ error: '服务器内部错误' });
  }
})
// 获取用户指定日期的打卡记录
app.get('/api/checkins/date', async (req, res) => {
  try {
    const { userId, date } = req.query;
    
    if (!userId || !date) {
      return res.status(400).json({ error: '用户ID和日期是必需的' });
    }
    
    // 查询指定日期的打卡记录
    const [checkins] = await connection.execute(
      'SELECT * FROM UserCheckins WHERE user_id = ? AND checkin_date = ?',
      [userId, date]
    );
    
    res.json({
      success: true,
      data: checkins.length > 0 ? checkins[0] : null
    });
  } catch (error) {
    console.error('获取打卡记录失败:', error);
    res.status(500).json({ error: '服务器内部错误' });
  }
});

// 搜索API
app.get('/api/search', async (req, res) => {
    const searchTerm = req.query.query;
    const currentUserId = req.query.userId || null; // 用于未来可能的个性化搜索或权限检查

    if (!searchTerm || searchTerm.trim() === '') {
        return res.status(400).json({ error: '搜索关键词不能为空' });
    }

    const searchQuery = `%${searchTerm}%`;

    try {
        // 1. 搜索用户`
        const userSearchQuery = `
            SELECT user_id, username, avatar_url, school_info, gender
            FROM Users
            WHERE username LIKE ?
            LIMIT 5;
        `;
        const [userResults] = await connection.execute(userSearchQuery, [searchQuery]);

        // 2. 搜索动态
        const postSearchQuery = `
            SELECT
                p.post_id, p.content, p.media_url, p.created_at, p.like_count, p.comment_count,
                u.user_id AS author_user_id,
                u.username AS author_username,
                u.gender AS author_gender,
                u.avatar_url AS author_avatar_url
            FROM Posts p
            JOIN Users u ON p.user_id = u.user_id
            WHERE p.content LIKE ?
            ORDER BY p.created_at DESC
            LIMIT 10;
        `;
        const [postResults] = await connection.execute(postSearchQuery, [searchQuery]);

        res.json({
            users: userResults,
            posts: postResults
        });

    } catch (error) {
        console.error('搜索API出错:', error);
        res.status(500).json({ error: '服务器内部错误，搜索失败' });
    }
});

// 启动服务器 保证数据库先启动
initDb().then(() => {
  app.listen(PORT, () => {
    console.log(`服务器正在运行在 http://localhost:${PORT}`);
  });
});   