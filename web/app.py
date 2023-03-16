import web    # 引入web.py库

# 表明访问的URL，这里表示的是所有响应，均由 class 对象 index 来响应
# 注：/(.*) 代表的是正则匹配url后面的所有路径，也就是响应任何请求
urls = (
     '/index/(.*)', 'index',
     '/model', 'model'
)

# 声明一个名叫app的“应用”
app = web.application(urls, globals())

# 表示 class 对象 index
# 传递参数：self，name（name指url路径/后面的内容）
class index:
    # 响应GET请求（声明函数）
    def GET(self,name):
        # 使用只读，二进制方式打开文件，读取到变量 index_text 中
        index_text = open('web/index.html','rb').read()
        # 输出变量 index_text 内的内容，也就是 index.html 内的HTML代码
        return index_text
    # 172.16.227.68:8080/model?name=aaa&age=12 获取到aaa12
class model:
    def POST(self):
        i = web.input(name=None)
        j = web.input(age=None)
        return i.name+j.age

# 当该.py文件被直接运行时，if __name__ == "__main__": 下的代码将被运行
# 当该.py文件作为模块被引入时，if __name__ == "__main__": 下的代码不会被运行
if __name__ == "__main__":
    # 运行这个服务器
    app.run()