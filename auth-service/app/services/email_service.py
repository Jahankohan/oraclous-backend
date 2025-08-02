import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.core.config import settings


async def send_email(recipient_email, subject, body):
    msg = MIMEMultipart()
    msg['From'] = settings.EMAIL_ADDRESS
    msg['To'] = recipient_email
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'plain'))

    # with smtplib.SMTP('smtp.gmail.com', 587) as server:
    #     server.starttls()
    #     server.login(settings.EMAIL_ADDRESS, settings.EMAIL_PASSWORD)
    #     server.send_message(msg)
