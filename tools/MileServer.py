# -*- coding: utf-8 -*-
import smtplib
from email.mime.text import MIMEText

class MileServer:
    #163邮箱服务器地址
    mail_host = 'smtp.163.com'  
    #163用户名
    mail_user = 'weijie_xu2021@163.com'  
    #密码(部分邮箱为授权码) 
    mail_pass = 'MBVYCPTKPVALOZIX' 
    #邮件发送方邮箱地址
    sender = 'weijie_xu2021@163.com'  
    receivers = ['1073645414@qq.com']
    message = None
    def __init__(self, mail_user=None, mail_passwd=None, sender=None):
        if mail_user is not None:
            self.mail_user = mail_user
        if mail_passwd is not None:
            self.mail_passwd = mail_passwd
        if sender is not None:
            self.sender = sender

    
    def setEmileContent(self, title, content):
        #邮件内容设置
        self.message = MIMEText(content, 'plain', 'utf-8')
        #邮件主题       
        self.message['Subject'] = title
        #发送方信息
        self.message['From'] = self.sender 
        #接受方信息     
        self.message['To'] = self.receivers[0]  
    
    def sendEmile(self, receivers=None):
        if receivers is not None:
            self.receivers = receivers
        #登录并发送邮件
        try:
            smtpObj = smtplib.SMTP() 
            #连接到服务器
            smtpObj.connect(self.mail_host, 25)
            #登录到服务器
            smtpObj.login(self.mail_user, self.mail_pass) 
            #发送
            smtpObj.sendmail(
                self.sender, self.receivers, self.message.as_string()) 
            #退出
            smtpObj.quit() 
            print('success')
        except smtplib.SMTPException as e:
            print('error',e) #打印错误

if __name__ == "__main__":
    server = MileServer()
    server.setEmileContent('Calculate Finished', 'calculate finished')
    server.sendEmile()








