import smtplib, ssl
from email import encoders
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart

# port = 465  # For SSL
# password = input("Type your password and press enter: ")

# # Create a secure SSL context
# context = ssl.create_default_context()

# sender_email = "jyserver00@gmail.com"
# receiver_email = "jyserver00@gmail.com"

######################################

# message = """\
# Subject: Hi there

# This message is sent from Python."""

######################################

# sender_email = "my@gmail.com"
# receiver_email = "your@gmail.com"
# password = input("Type your password and press enter:")

# message = MIMEMultipart("alternative")
# message["Subject"] = "multipart test"
# message["From"] = sender_email
# message["To"] = receiver_email

# # Create the plain-text and HTML version of your message
# text = """\
# Hi,
# How are you?
# Real Python has many great tutorials:
# www.realpython.com"""
# html = """\
# <html>
#   <body>
#     <p>Hi,<br>
#        How are you?<br>
#        <a href="http://www.realpython.com">Real Python</a> 
#        has many great tutorials.
#     </p>
#   </body>
# </html>
# """

# # Turn these into plain/html MIMEText objects
# part1 = MIMEText(text, "plain")
# part2 = MIMEText(html, "html")

# # Add HTML/plain-text parts to MIMEMultipart message
# # The email client will try to render the last part first
# message.attach(part1)
# message.attach(part2)


######################################

# # Create a multipart message and set headers
# subject = "An email with attachment from Python"
# body = "This is an email with attachment sent from Python"

# message = MIMEMultipart()
# message["From"] = sender_email
# message["To"] = receiver_email
# message["Subject"] = subject
# message["Bcc"] = receiver_email  # Recommended for mass emails

# # Add body to email
# message.attach(MIMEText(body, "plain"))

# filename = "pytorch-internals.pdf"  # In same directory as script

# # Open PDF file in binary mode
# with open(filename, "rb") as attachment:
#     # Add file as application/octet-stream
#     # Email client can usually download this automatically as attachment
#     part = MIMEBase("application", "octet-stream")
#     part.set_payload(attachment.read())

# # Encode file in ASCII characters to send by email    
# encoders.encode_base64(part)

# # Add header as key/value pair to attachment part
# part.add_header(
#     "Content-Disposition",
#     f"attachment; filename= {filename}",
# )

# # Add attachment to message and convert message to string
# message.attach(part)
# text = message.as_string()

######################################


# with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
#     server.login("jyserver00@gmail.com", password)
#     # server.sendmail(sender_email, receiver_email, message)
#     # server.sendmail(sender_email, receiver_email, message.as_string())
#     server.sendmail(sender_email, receiver_email, text)




# 2 
# import smtplib, ssl

# smtp_server = "smtp.gmail.com"
# port = 587  # For starttls
# sender_email = "my@gmail.com"
# password = input("Type your password and press enter: ")

# # Create a secure SSL context
# context = ssl.create_default_context()

# # Try to log in to server and send email
# try:
#     server = smtplib.SMTP(smtp_server,port)
#     server.ehlo() # Can be omitted
#     server.starttls(context=context) # Secure the connection
#     server.ehlo() # Can be omitted
#     server.login(sender_email, password)
#     # TODO: Send email here
# except Exception as e:
#     # Print any error messages to stdout
#     print(e)
# finally:
#     server.quit() 



######################################


import yagmail

receiver = "jyserver00@gmail.com"
body = "Hello there from Yagmail"
filename = "pytorch-internals.pdf"

yag = yagmail.SMTP("jyserver00@gmail.com")
yag.send(
    to=receiver,
    subject="Yagmail test with attachment",
    contents=body, 
    attachments=filename,
)