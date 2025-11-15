/**
 * Shared table pagination utilities
 * 
 * This file contains common pagination functions that prevent going beyond available pages.
 * Include this in all dashboards to maintain consistent pagination behavior.
 * 
 * Usage:
 *   1. Include this script: <script src="{{ url_for('static', filename='js/table-pagination.js') }}"></script>
 *   2. Use the helper functions in your pagination handlers
 */

/**
 * Check if next page is available (server-side pagination)
 * @param {number} currentPage - Current page number
 * @param {number} total - Total number of items
 * @param {number} pageSize - Items per page
 * @returns {boolean} - True if next page is available
 */
function canGoToNextPage(currentPage, total, pageSize) {
    const maxPage = Math.ceil(total / pageSize);
    return currentPage < maxPage;
}

/**
 * Check if previous page is available
 * @param {number} currentPage - Current page number
 * @returns {boolean} - True if previous page is available
 */
function canGoToPrevPage(currentPage) {
    return currentPage > 1;
}

/**
 * Get maximum page number
 * @param {number} total - Total number of items
 * @param {number} pageSize - Items per page
 * @returns {number} - Maximum page number
 */
function getMaxPage(total, pageSize) {
    return Math.ceil(total / pageSize);
}

/**
 * Format count display as "Showing X-Y of Z"
 * @param {number} startIndex - Zero-based start index
 * @param {number} endIndex - Zero-based end index (exclusive)
 * @param {number} total - Total number of items
 * @returns {string} - Formatted count display
 */
function formatCountDisplay(startIndex, endIndex, total) {
    const displayStart = total > 0 ? startIndex + 1 : 0;
    const displayEnd = total > 0 ? endIndex : 0;
    return total > 0
        ? `Showing ${displayStart}-${displayEnd} of ${total}`
        : `Showing 0-0 of 0`;
}

/**
 * Create a safe next page handler that prevents going beyond available pages
 * @param {Object} config - Configuration object
 * @param {Function} config.onNext - Callback when next page is clicked (receives new page number)
 * @param {Function} config.getCurrentPage - Function that returns current page number
 * @param {Function} config.getTotal - Function that returns total count
 * @param {Function} config.getPageSize - Function that returns page size
 * @param {boolean} config.serverSide - Whether using server-side pagination (default: true)
 * @returns {Function} - Next page handler function
 */
function createSafeNextPageHandler(config) {
    return function() {
        const currentPage = config.getCurrentPage();
        const total = config.getTotal();
        const pageSize = config.getPageSize();
        
        if (canGoToNextPage(currentPage, total, pageSize)) {
            const newPage = currentPage + 1;
            if (config.onNext) {
                config.onNext(newPage);
            }
        }
    };
}

/**
 * Create a safe previous page handler
 * @param {Object} config - Configuration object
 * @param {Function} config.onPrev - Callback when previous page is clicked (receives new page number)
 * @param {Function} config.getCurrentPage - Function that returns current page number
 * @returns {Function} - Previous page handler function
 */
function createSafePrevPageHandler(config) {
    return function() {
        const currentPage = config.getCurrentPage();
        
        if (canGoToPrevPage(currentPage)) {
            const newPage = currentPage - 1;
            if (config.onPrev) {
                config.onPrev(newPage);
            }
        }
    };
}
