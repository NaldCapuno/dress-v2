/**
 * Shared table pagination functions
 * Include this file in all dashboards to use consistent pagination logic
 */

/**
 * Create pagination handlers for a table
 * This function creates reusable pagination handlers that prevent going beyond available pages
 * 
 * Usage in dashboard:
 *   // Initialize pagination state
 *   window.violationsPagination = { page: 1, pageSize: 10, total: 0 };
 *   
 *   // Create handlers
 *   const violationsPagination = createPaginationHandlers({
 *     instanceId: 'violationsPagination',
 *     dataLoader: loadViolations,  // Function that loads data (uses page/pageSize from instance)
 *     renderFunction: renderViolationsTable,
 *     pageSizeId: 'page-size',
 *     serverSide: true,
 *     getTotalFunction: () => violationsTotal  // For client-side: function to get current total
 *   });
 *   
 *   // Expose as global functions for onclick handlers
 *   window.prevPage = () => violationsPagination.prevPage();
 *   window.nextPage = () => violationsPagination.nextPage();
 *   window.changePageSize = () => violationsPagination.changePageSize();
 */
function createPaginationHandlers(config) {
    const {
        instanceId,
        dataLoader,
        renderFunction,
        getTotalFunction,
        pageSizeId,
        serverSide = true
    } = config;
    
    // Ensure instance exists
    if (!window[instanceId]) {
        window[instanceId] = {
            page: 1,
            pageSize: 10,
            total: 0
        };
    }
    
    const instance = window[instanceId];
    
    // Helper to call dataLoader with current page/pageSize
    const callDataLoader = () => {
        if (dataLoader) {
            // Update global page/pageSize variables if they exist (for backward compatibility)
            if (typeof page !== 'undefined') page = instance.page;
            if (typeof pageSize !== 'undefined') pageSize = instance.pageSize;
            
            const result = dataLoader();
            if (result && result.then) {
                return result.then(() => {
                    if (renderFunction) renderFunction();
                });
            } else if (renderFunction) {
                renderFunction();
            }
        } else if (renderFunction) {
            renderFunction();
        }
    };
    
    return {
        prevPage: function() {
            if (instance.page > 1) {
                instance.page--;
                callDataLoader();
            }
        },
        
        nextPage: function() {
            let maxPage;
            
            if (serverSide) {
                maxPage = Math.ceil(instance.total / instance.pageSize);
            } else {
                const currentTotal = getTotalFunction ? getTotalFunction() : instance.total;
                maxPage = Math.ceil(currentTotal / instance.pageSize);
            }
            
            if (instance.page < maxPage) {
                instance.page++;
                callDataLoader();
            }
        },
        
        changePageSize: function() {
            const select = document.getElementById(pageSizeId);
            if (select) {
                instance.pageSize = parseInt(select.value);
                instance.page = 1;
                callDataLoader();
            }
        }
    };
}

/**
 * Format count display as "Showing X-Y of Z"
 */
function formatCountDisplay(startIndex, endIndex, total) {
    const displayStart = total > 0 ? startIndex + 1 : 0;
    const displayEnd = total > 0 ? endIndex : 0;
    return total > 0
        ? `Showing ${displayStart}-${displayEnd} of ${total}`
        : `Showing 0-0 of 0`;
}

