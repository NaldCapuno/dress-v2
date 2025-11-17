"""
Email templates for DRESS system notifications.
Contains HTML email templates with mobile-responsive design.
"""


def generate_violation_email_body(student_name, violation_datetime, strike_num, offense_line, violation_history, image_cid=None, logo_base64=None, logo_cid='dress_logo'):
    """
    Generate HTML email body for dress code violation notification.
    Mobile-responsive design with inline CSS matching web app color scheme.
    
    Args:
        student_name (str): Name of the student
        violation_datetime (str): Date and time of the violation
        strike_num (int): Current strike number (1-3)
        offense_line (str): Text description of the offense (e.g., "1st Offense")
        violation_history (str): Formatted list of previous violations
        image_cid (str, optional): Content-ID (CID) for inline image attachment
    
    Returns:
        str: HTML formatted email body
    """
    # Format violation history as HTML list (matching web app colors)
    if violation_history and violation_history != 'No history available':
        history_items = violation_history.split('\n')
        formatted_history = '<ul style="margin: 10px 0; padding-left: 20px;">'
        for item in history_items:
            if item.strip():
                formatted_history += f'<li style="margin: 5px 0; color: #374151; font-size: 14px;">{item.strip()}</li>'
        formatted_history += '</ul>'
    else:
        formatted_history = '<p style="color: #9ca3af; font-style: italic; font-size: 14px;">No history available</p>'
    
    # Determine strike color based on number (matching web app colors)
    if strike_num == 1:
        strike_color = '#f59e0b'  # Warning (matches web app)
    elif strike_num == 2:
        strike_color = '#ef4444'  # Error (matches web app)
    else:
        strike_color = '#e55100'  # Accent-dark (matches web app)
    
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dress Code Violation Notification</title>
</head>
<body style="margin: 0; padding: 0; font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif; background-color: #f8fafc;">
    <table role="presentation" style="width: 100%; border-collapse: collapse; background-color: #f8fafc; padding: 20px 0;">
        <tr>
            <td align="center" style="padding: 20px 10px;">
                <table role="presentation" style="max-width: 600px; width: 100%; border-collapse: collapse; background-color: #ffffff; border-radius: 8px; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);">
                    <!-- Header -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #2ca9e1 0%, #1e7bb8 100%); padding: 30px 20px; text-align: center; border-radius: 8px 8px 0 0;">
                            <h1 style="margin: 0; color: #ffffff; font-size: 24px; font-weight: 600; letter-spacing: 0.5px;">
                                DRESS CODE VIOLATION
                            </h1>
                            <p style="margin: 10px 0 0 0; color: #ffffff; font-size: 14px; opacity: 0.95;">
                                Dress-code Recognition Surveillance System
                            </p>
                        </td>
                    </tr>
                    
                    <!-- Content -->
                    <tr>
                        <td style="padding: 30px 20px;">
                            <p style="margin: 0 0 15px 0; color: #374151; font-size: 16px; line-height: 1.6;">
                                Dear <strong style="color: #2ca9e1;">{student_name}</strong>,
                            </p>
                            <p style="margin: 0 0 20px 0; color: #4b5563; font-size: 15px; line-height: 1.6;">
                                This is to inform you that the DRESS (Dress-code Recognition Surveillance System) detected a dress code violation on your part on <strong style="color: #374151;">{violation_datetime}</strong>.
                            </p>
                            <p style="margin: 0 0 25px 0; color: #4b5563; font-size: 15px; line-height: 1.6;">
                                Please remember that following the university dress code is part of maintaining discipline and professionalism. We ask you to comply with the proper uniform prescribed by the University, as stated in the Student Handbook, on your next visit.
                            </p>
                            
                            <!-- Violation Details Box -->
                            <div style="background-color: #f8fafc; border-left: 4px solid {strike_color}; padding: 20px; margin: 25px 0; border-radius: 4px;">
                                <h2 style="margin: 0 0 15px 0; color: #1f2937; font-size: 18px; font-weight: 600;">
                                    Violation Details
                                </h2>
                                <table role="presentation" style="width: 100%; border-collapse: collapse;">
                                    <tr>
                                        <td style="padding: 8px 0; color: #6b7280; font-size: 14px; width: 50%;">Current Strike Count:</td>
                                        <td style="padding: 8px 0; color: {strike_color}; font-size: 16px; font-weight: 600;">{strike_num} of 3</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 8px 0; color: #6b7280; font-size: 14px;">Your Current Offense:</td>
                                        <td style="padding: 8px 0; color: #374151; font-size: 14px; font-weight: 500;">{offense_line}</td>
                                    </tr>
                                </table>
                                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e5e7eb;">
                                    <p style="margin: 0 0 10px 0; color: #6b7280; font-size: 14px; font-weight: 500;">Recorded Violations:</p>
                                    {violation_history}
                                </div>
                            </div>
                            
                            <!-- Proof Image -->
                            {proof_image_section}
                            
                            <!-- Guidelines Box -->
                            <div style="background-color: #fff7ed; border: 1px solid #f25a04; padding: 20px; margin: 25px 0; border-radius: 4px;">
                                <h2 style="margin: 0 0 15px 0; color: #1f2937; font-size: 18px; font-weight: 600;">
                                    University Guidelines
                                </h2>
                                <ul style="margin: 0; padding-left: 20px; color: #4b5563; font-size: 14px; line-height: 1.8;">
                                    <li style="margin: 5px 0;"><strong style="color: #f25a04;">1st Offense</strong> – Warning</li>
                                    <li style="margin: 5px 0;"><strong style="color: #f25a04;">2nd Offense</strong> – 5-day suspension</li>
                                    <li style="margin: 5px 0;"><strong style="color: #f25a04;">3rd Offense</strong> – 2-week to 1-month suspension</li>
                                </ul>
                            </div>
                            
                            <!-- Action Required -->
                            <div style="background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 15px 20px; margin: 25px 0; border-radius: 4px;">
                                <p style="margin: 0; color: #991b1b; font-size: 15px; font-weight: 600;">
                                    ⚠️ Action Required
                                </p>
                                <p style="margin: 10px 0 0 0; color: #7f1d1d; font-size: 14px; line-height: 1.6;">
                                    Please report to the Guidance Office to address this matter and complete the required procedures.
                                </p>
                            </div>
                            
                            <p style="margin: 25px 0 0 0; color: #4b5563; font-size: 15px; line-height: 1.6;">
                                Thank you for your cooperation.
                            </p>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td style="background-color: #f8fafc; padding: 25px 20px; text-align: center; border-radius: 0 0 8px 8px; border-top: 1px solid #e5e7eb;">
                            <p style="margin: 0 0 10px 0; color: #374151; font-size: 15px; font-weight: 500;">
                                Respectfully,
                            </p>
                            <p style="margin: 0 0 5px 0; color: #2ca9e1; font-size: 14px; font-weight: 600;">
                                DRESS Monitoring Team
                            </p>
                            <p style="margin: 0; color: #6b7280; font-size: 13px;">
                                Palawan State University
                            </p>
                            <p style="margin: 20px 0 0 0; color: #9ca3af; font-size: 12px; font-style: italic;">
                                This is an automated notification. Please do not reply to this email.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""
    
    # Generate proof image section if image CID is provided
    if image_cid:
        proof_image_section = f"""
                            <div style="background-color: #f8fafc; padding: 20px; margin: 25px 0; border-radius: 4px; border: 1px solid #e5e7eb;">
                                <h2 style="margin: 0 0 15px 0; color: #1f2937; font-size: 18px; font-weight: 600;">
                                    Proof of Violation
                                </h2>
                                <div style="text-align: center; margin: 15px 0;">
                                    <img src="cid:{image_cid}" alt="Violation Proof Image" style="max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); border: 1px solid #e5e7eb;" />
                                </div>
                                <p style="margin: 10px 0 0 0; color: #6b7280; font-size: 12px; text-align: center; font-style: italic;">
                                    Proof image attached to this email
                                </p>
                            </div>"""
    else:
        proof_image_section = ""
    
    return html_template.format(
        student_name=student_name,
        violation_datetime=violation_datetime,
        strike_num=strike_num,
        offense_line=offense_line,
        violation_history=formatted_history,
        strike_color=strike_color,
        proof_image_section=proof_image_section,
        logo_section=""
    )


def generate_followup_email_body(student_name, first_notice_date, violation_datetime, strike_num, offense_line, violation_history, image_cid=None, logo_base64=None, logo_cid='dress_logo'):
    """
    Generate HTML email body for dress code violation follow-up notification.
    Sent after 3 days if violation is still not resolved.
    Mobile-responsive design with inline CSS matching web app color scheme.
    
    Args:
        student_name (str): Name of the student
        first_notice_date (str): Date of the first notification
        violation_datetime (str): Date and time of the violation
        strike_num (int): Current strike number (1-3)
        offense_line (str): Text description of the offense (e.g., "1st Offense")
        violation_history (str): Formatted list of previous violations
        image_cid (str, optional): Content-ID (CID) for inline image attachment
    
    Returns:
        str: HTML formatted email body
    """
    # Format violation history as HTML list (matching web app colors)
    if violation_history and violation_history != 'No history available':
        history_items = violation_history.split('\n')
        formatted_history = '<ul style="margin: 10px 0; padding-left: 20px;">'
        for item in history_items:
            if item.strip():
                formatted_history += f'<li style="margin: 5px 0; color: #374151; font-size: 14px;">{item.strip()}</li>'
        formatted_history += '</ul>'
    else:
        formatted_history = '<p style="color: #9ca3af; font-style: italic; font-size: 14px;">No history available</p>'
    
    # Determine strike color based on number (matching web app colors)
    if strike_num == 1:
        strike_color = '#f59e0b'  # Warning (matches web app)
    elif strike_num == 2:
        strike_color = '#ef4444'  # Error (matches web app)
    else:
        strike_color = '#e55100'  # Accent-dark (matches web app)
    
    # Generate image attachment text
    image_attachment_text = '<p style="margin: 10px 0 0 0; color: #6b7280; font-size: 12px; font-style: italic;">Proof image attached to this email</p>' if image_cid else ''
    
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dress Code Violation Follow-Up Notice</title>
</head>
<body style="margin: 0; padding: 0; font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif; background-color: #f8fafc;">
    <table role="presentation" style="width: 100%; border-collapse: collapse; background-color: #f8fafc; padding: 20px 0;">
        <tr>
            <td align="center" style="padding: 20px 10px;">
                <table role="presentation" style="max-width: 600px; width: 100%; border-collapse: collapse; background-color: #ffffff; border-radius: 8px; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);">
                    <!-- Header -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); padding: 30px 20px; text-align: center; border-radius: 8px 8px 0 0;">
                            <h1 style="margin: 0; color: #ffffff; font-size: 24px; font-weight: 600; letter-spacing: 0.5px;">
                                DRESS CODE VIOLATION FOLLOW-UP NOTICE
                            </h1>
                            <p style="margin: 10px 0 0 0; color: #ffffff; font-size: 14px; opacity: 0.95;">
                                Dress-code Recognition Surveillance System
                            </p>
                        </td>
                    </tr>
                    
                    <!-- Content -->
                    <tr>
                        <td style="padding: 30px 20px;">
                            <p style="margin: 0 0 15px 0; color: #374151; font-size: 16px; line-height: 1.6;">
                                Dear <strong style="color: #2ca9e1;">{student_name}</strong>,
                            </p>
                            <p style="margin: 0 0 20px 0; color: #4b5563; font-size: 15px; line-height: 1.6;">
                                This is a follow-up to the DRESS (Dress-code Recognition Surveillance System) notification sent to you. Our records show that the dress code violation detected on <strong style="color: #374151;">{violation_datetime}</strong> has not yet been addressed.
                            </p>
                            <p style="margin: 0 0 25px 0; color: #4b5563; font-size: 15px; line-height: 1.6;">
                                Following the university dress code is an important part of maintaining discipline and professionalism. We remind you to comply with the proper uniform prescribed by the University, as stated in the Student Handbook, on your next visit.
                            </p>
                            
                            <!-- Violation Details Box -->
                            <div style="background-color: #f8fafc; border-left: 4px solid {strike_color}; padding: 20px; margin: 25px 0; border-radius: 4px;">
                                <h2 style="margin: 0 0 15px 0; color: #1f2937; font-size: 18px; font-weight: 600;">
                                    VIOLATION DETAILS
                                </h2>
                                <table role="presentation" style="width: 100%; border-collapse: collapse;">
                                    <tr>
                                        <td style="padding: 8px 0; color: #6b7280; font-size: 14px; width: 50%;">Current Strike Count:</td>
                                        <td style="padding: 8px 0; color: {strike_color}; font-size: 16px; font-weight: 600;">{strike_num} of 3</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 8px 0; color: #6b7280; font-size: 14px;">Your Recorded Offense:</td>
                                        <td style="padding: 8px 0; color: #374151; font-size: 14px; font-weight: 500;">{offense_line}</td>
                                    </tr>
                                </table>
                                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e5e7eb;">
                                    <p style="margin: 0 0 10px 0; color: #6b7280; font-size: 14px; font-weight: 500;">Previously Recorded Violations:</p>
                                    {violation_history}
                                    {image_attachment_text}
                                </div>
                            </div>
                            
                            <!-- Proof Image -->
                            {proof_image_section}
                            
                            <!-- Guidelines Box -->
                            <div style="background-color: #fff7ed; border: 1px solid #f25a04; padding: 20px; margin: 25px 0; border-radius: 4px;">
                                <h2 style="margin: 0 0 15px 0; color: #1f2937; font-size: 18px; font-weight: 600;">
                                    UNIVERSITY GUIDELINES
                                </h2>
                                <ul style="margin: 0; padding-left: 20px; color: #4b5563; font-size: 14px; line-height: 1.8;">
                                    <li style="margin: 5px 0;"><strong style="color: #f25a04;">1st Offense</strong> – Warning</li>
                                    <li style="margin: 5px 0;"><strong style="color: #f25a04;">2nd Offense</strong> – 5-day suspension</li>
                                    <li style="margin: 5px 0;"><strong style="color: #f25a04;">3rd Offense</strong> – 2-week to 1-month suspension</li>
                                </ul>
                            </div>
                            
                            <!-- Action Required -->
                            <div style="background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 15px 20px; margin: 25px 0; border-radius: 4px;">
                                <p style="margin: 0; color: #991b1b; font-size: 15px; font-weight: 600;">
                                    ⚠️ ACTION REQUIRED
                                </p>
                                <p style="margin: 10px 0 0 0; color: #7f1d1d; font-size: 14px; line-height: 1.6;">
                                    Please report to the Guidance Office as soon as possible to settle this matter. Continued failure to respond may affect the sanction applied to your case.
                                </p>
                            </div>
                            
                            <p style="margin: 25px 0 0 0; color: #4b5563; font-size: 15px; line-height: 1.6;">
                                Thank you for your immediate attention.
                            </p>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td style="background-color: #f8fafc; padding: 25px 20px; text-align: center; border-radius: 0 0 8px 8px; border-top: 1px solid #e5e7eb;">
                            <p style="margin: 0 0 10px 0; color: #374151; font-size: 15px; font-weight: 500;">
                                Respectfully,
                            </p>
                            <p style="margin: 0 0 5px 0; color: #2ca9e1; font-size: 14px; font-weight: 600;">
                                DRESS Monitoring Team
                            </p>
                            <p style="margin: 0; color: #6b7280; font-size: 13px;">
                                Palawan State University
                            </p>
                            <p style="margin: 20px 0 0 0; color: #9ca3af; font-size: 12px; font-style: italic;">
                                This is an automated notification. Please do not reply to this email.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""
    
    # Generate proof image section if image CID is provided
    if image_cid:
        proof_image_section = f"""
                            <div style="background-color: #f8fafc; padding: 20px; margin: 25px 0; border-radius: 4px; border: 1px solid #e5e7eb;">
                                <h2 style="margin: 0 0 15px 0; color: #1f2937; font-size: 18px; font-weight: 600;">
                                    Proof of Violation
                                </h2>
                                <div style="text-align: center; margin: 15px 0;">
                                    <img src="cid:{image_cid}" alt="Violation Proof Image" style="max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); border: 1px solid #e5e7eb;" />
                                </div>
                                {image_attachment_text}
                            </div>"""
    else:
        proof_image_section = ""
    
    return html_template.format(
        student_name=student_name,
        first_notice_date=first_notice_date,
        violation_datetime=violation_datetime,
        strike_num=strike_num,
        offense_line=offense_line,
        violation_history=formatted_history,
        strike_color=strike_color,
        proof_image_section=proof_image_section,
        image_attachment_text=image_attachment_text,
        logo_section=""
    )


def generate_password_reset_email_body(username, reset_code, include_username=False, logo_base64=None, logo_cid='dress_logo'):
    """
    Generate HTML email body for password reset code notification.
    Mobile-responsive design with inline CSS matching web app color scheme.
    
    Args:
        username (str): Username of the admin requesting password reset
        reset_code (str): 6-digit reset code
        include_username (bool): If True, include username in the email (for when user forgot username)
    
    Returns:
        str: HTML formatted email body
    """
    # Username section (only shown if include_username is True)
    if include_username:
        username_section = f"""
                            <!-- Username Box -->
                            <div style="background-color: #fff7ed; border: 2px solid #f25a04; padding: 20px; margin: 25px 0; border-radius: 8px; text-align: center;">
                                <p style="margin: 0 0 10px 0; color: #6b7280; font-size: 14px; font-weight: 500;">
                                    Your Username:
                                </p>
                                <p style="margin: 0; color: #f25a04; font-size: 24px; font-weight: 700; font-family: 'Courier New', monospace;">
                                    {username}
                                </p>
                            </div>
"""
    else:
        username_section = ""
    
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Password Reset Code</title>
</head>
<body style="margin: 0; padding: 0; font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif; background-color: #f8fafc;">
    <table role="presentation" style="width: 100%; border-collapse: collapse; background-color: #f8fafc; padding: 20px 0;">
        <tr>
            <td align="center" style="padding: 20px 10px;">
                <table role="presentation" style="max-width: 600px; width: 100%; border-collapse: collapse; background-color: #ffffff; border-radius: 8px; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);">
                    <!-- Header -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #2ca9e1 0%, #1e7bb8 100%); padding: 30px 20px; text-align: center; border-radius: 8px 8px 0 0;">
                            <h1 style="margin: 0; color: #ffffff; font-size: 24px; font-weight: 600; letter-spacing: 0.5px;">
                                PASSWORD RESET REQUEST
                            </h1>
                            <p style="margin: 10px 0 0 0; color: #ffffff; font-size: 14px; opacity: 0.95;">
                                Dress-code Recognition Surveillance System
                            </p>
                        </td>
                    </tr>
                    
                    <!-- Content -->
                    <tr>
                        <td style="padding: 30px 20px;">
                            <p style="margin: 0 0 15px 0; color: #374151; font-size: 16px; line-height: 1.6;">
                                Hello <strong style="color: #2ca9e1;">{username}</strong>,
                            </p>
                            <p style="margin: 0 0 20px 0; color: #4b5563; font-size: 15px; line-height: 1.6;">
                                We received a request to reset your password for your DRESS admin account. Use the code below to reset your password:
                            </p>
                            {username_section}
                            <!-- Reset Code Box -->
                            <div style="background-color: #f8fafc; border: 2px solid #2ca9e1; padding: 30px; margin: 25px 0; border-radius: 8px; text-align: center;">
                                <p style="margin: 0 0 10px 0; color: #6b7280; font-size: 14px; font-weight: 500;">
                                    Your Password Reset Code:
                                </p>
                                <p style="margin: 0; color: #2ca9e1; font-size: 36px; font-weight: 700; letter-spacing: 8px; font-family: 'Courier New', monospace;">
                                    {reset_code}
                                </p>
                            </div>
                            
                            <p style="margin: 20px 0 0 0; color: #4b5563; font-size: 15px; line-height: 1.6;">
                                This code will expire in <strong style="color: #ef4444;">15 minutes</strong>. If you did not request this password reset, please ignore this email or contact the system administrator.
                            </p>
                            
                            <!-- Security Notice -->
                            <div style="background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 15px 20px; margin: 25px 0; border-radius: 4px;">
                                <p style="margin: 0; color: #991b1b; font-size: 14px; font-weight: 600;">
                                    🔒 Security Notice
                                </p>
                                <p style="margin: 10px 0 0 0; color: #7f1d1d; font-size: 13px; line-height: 1.6;">
                                    Never share this code with anyone. DRESS staff will never ask for your password reset code.
                                </p>
                            </div>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td style="background-color: #f8fafc; padding: 25px 20px; text-align: center; border-radius: 0 0 8px 8px; border-top: 1px solid #e5e7eb;">
                            <p style="margin: 0 0 10px 0; color: #374151; font-size: 15px; font-weight: 500;">
                                Respectfully,
                            </p>
                            <p style="margin: 0 0 5px 0; color: #2ca9e1; font-size: 14px; font-weight: 600;">
                                DRESS System
                            </p>
                            <p style="margin: 0; color: #6b7280; font-size: 13px;">
                                Palawan State University
                            </p>
                            <p style="margin: 20px 0 0 0; color: #9ca3af; font-size: 12px; font-style: italic;">
                                This is an automated notification. Please do not reply to this email.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""
    
    return html_template.format(
        username=username,
        reset_code=reset_code,
        username_section=username_section,
        logo_section=""
    )