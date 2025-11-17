/**
 * Toast Notification System
 * Reusable toast notification function for displaying messages
 * 
 * Usage:
 *   showToast('Message here', 'success');  // success, error, or info
 */

function showToast(message, type = 'info') {
    // Get or create toast container
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }
    
    // Create unique ID for this toast
    const toastId = 'toast-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    
    // Create toast element
    const toast = document.createElement('div');
    toast.id = toastId;
    toast.style.cssText = 'padding: 15px 20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); min-width: 250px; max-width: 400px; animation: slideInRight 0.3s ease-out; pointer-events: auto; display: flex; align-items: center; gap: 10px;';
    
    // Create icon
    const icon = document.createElement('i');
    icon.style.cssText = 'font-size: 20px; flex-shrink: 0;';
    
    // Create message
    const messageSpan = document.createElement('span');
    messageSpan.textContent = message;
    messageSpan.style.cssText = 'flex: 1; font-size: 14px; font-weight: 500;';
    
    // Set colors and icon based on type
    if (type === 'success') {
        toast.style.backgroundColor = '#d4edda';
        toast.style.color = '#155724';
        toast.style.border = '1px solid #c3e6cb';
        icon.className = 'fas fa-check-circle';
        icon.style.color = '#155724';
    } else if (type === 'error') {
        toast.style.backgroundColor = '#f8d7da';
        toast.style.color = '#721c24';
        toast.style.border = '1px solid #f5c6cb';
        icon.className = 'fas fa-exclamation-circle';
        icon.style.color = '#721c24';
    } else {
        toast.style.backgroundColor = '#d1ecf1';
        toast.style.color = '#0c5460';
        toast.style.border = '1px solid #bee5eb';
        icon.className = 'fas fa-info-circle';
        icon.style.color = '#0c5460';
    }
    
    toast.appendChild(icon);
    toast.appendChild(messageSpan);
    toastContainer.appendChild(toast);
    
    // Function to remove toast
    function removeToast() {
        const toastElement = document.getElementById(toastId);
        if (!toastElement) return;
        toastElement.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => {
            if (toastElement && toastElement.parentNode) {
                toastElement.parentNode.removeChild(toastElement);
            }
        }, 300);
    }
    
    // Auto-remove after 3 seconds
    setTimeout(removeToast, 3000);
}

