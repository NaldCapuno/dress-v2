/**
 * Shared table utilities for pagination, sorting, and filtering
 * Used across all dashboards
 */

// Pagination utilities
const TableUtils = {
    /**
     * Initialize pagination for a table
     * @param {Object} config - Configuration object
     */
    initPagination(config) {
        const {
            pageSizeId,
            pageId,
            countDisplayId,
            prevBtnId,
            nextBtnId,
            dataLoader, // Function that loads data and returns {rows, total}
            renderFunction, // Function to render the table
            getTotalFunction, // Function to get total count (for client-side)
            serverSide = true // Whether pagination is server-side or client-side
        } = config;
        
        let page = 1;
        let pageSize = 10;
        let total = 0;
        
        // Store in window for global access
        const instanceId = config.instanceId || 'tablePagination';
        window[instanceId] = {
            page,
            pageSize,
            total,
            setTotal: (t) => { total = t; },
            getPage: () => page,
            getPageSize: () => pageSize,
            getTotal: () => total
        };
        
        // Page size change handler
        const pageSizeEl = document.getElementById(pageSizeId);
        if (pageSizeEl) {
            pageSizeEl.addEventListener('change', function() {
                pageSize = parseInt(this.value);
                page = 1;
                window[instanceId].page = page;
                window[instanceId].pageSize = pageSize;
                
                if (serverSide && dataLoader) {
                    dataLoader(page, pageSize).then(result => {
                        if (result) {
                            total = result.total || 0;
                            window[instanceId].total = total;
                            if (renderFunction) renderFunction();
                            updatePageDisplay();
                        }
                    });
                } else if (renderFunction) {
                    renderFunction();
                    updatePageDisplay();
                }
            });
        }
        
        // Previous page handler
        const prevBtn = document.getElementById(prevBtnId);
        if (prevBtn) {
            prevBtn.addEventListener('click', function() {
                if (page > 1) {
                    page--;
                    window[instanceId].page = page;
                    
                    if (serverSide && dataLoader) {
                        dataLoader(page, pageSize).then(result => {
                            if (result) {
                                total = result.total || 0;
                                window[instanceId].total = total;
                                if (renderFunction) renderFunction();
                                updatePageDisplay();
                            }
                        });
                    } else if (renderFunction) {
                        renderFunction();
                        updatePageDisplay();
                    }
                }
            });
        }
        
        // Next page handler
        const nextBtn = document.getElementById(nextBtnId);
        if (nextBtn) {
            nextBtn.addEventListener('click', function() {
                let maxPage;
                
                if (serverSide) {
                    maxPage = Math.ceil(total / pageSize);
                } else {
                    const currentTotal = getTotalFunction ? getTotalFunction() : total;
                    maxPage = Math.ceil(currentTotal / pageSize);
                }
                
                if (page < maxPage) {
                    page++;
                    window[instanceId].page = page;
                    
                    if (serverSide && dataLoader) {
                        dataLoader(page, pageSize).then(result => {
                            if (result) {
                                total = result.total || 0;
                                window[instanceId].total = total;
                                if (renderFunction) renderFunction();
                                updatePageDisplay();
                            }
                        });
                    } else if (renderFunction) {
                        renderFunction();
                        updatePageDisplay();
                    }
                }
            });
        }
        
        function updatePageDisplay() {
            const pageEl = document.getElementById(pageId);
            if (pageEl) {
                pageEl.textContent = page;
            }
        }
        
        // Return instance for manual control
        return {
            setPage: (p) => { page = p; window[instanceId].page = page; },
            setPageSize: (ps) => { pageSize = ps; window[instanceId].pageSize = pageSize; },
            setTotal: (t) => { total = t; window[instanceId].total = t; },
            getPage: () => page,
            getPageSize: () => pageSize,
            getTotal: () => total
        };
    },
    
    /**
     * Format count display as "Showing X-Y of Z"
     */
    formatCountDisplay(startIndex, endIndex, total) {
        const displayStart = total > 0 ? startIndex + 1 : 0;
        const displayEnd = total > 0 ? endIndex : 0;
        return total > 0
            ? `Showing ${displayStart}-${displayEnd} of ${total}`
            : `Showing 0-0 of 0`;
    },
    
    /**
     * Create pagination functions that can be called from onclick handlers
     */
    createPaginationHandlers(config) {
        const {
            instanceId = 'tablePagination',
            dataLoader,
            renderFunction,
            getTotalFunction,
            serverSide = true
        } = config;
        
        return {
            prevPage: function() {
                const instance = window[instanceId];
                if (!instance) return;
                
                if (instance.page > 1) {
                    instance.page--;
                    
                    if (serverSide && dataLoader) {
                        dataLoader(instance.page, instance.pageSize).then(result => {
                            if (result) {
                                instance.total = result.total || 0;
                                if (renderFunction) renderFunction();
                            }
                        });
                    } else if (renderFunction) {
                        renderFunction();
                    }
                }
            },
            
            nextPage: function() {
                const instance = window[instanceId];
                if (!instance) return;
                
                let maxPage;
                if (serverSide) {
                    maxPage = Math.ceil(instance.total / instance.pageSize);
                } else {
                    const currentTotal = getTotalFunction ? getTotalFunction() : instance.total;
                    maxPage = Math.ceil(currentTotal / instance.pageSize);
                }
                
                if (instance.page < maxPage) {
                    instance.page++;
                    
                    if (serverSide && dataLoader) {
                        dataLoader(instance.page, instance.pageSize).then(result => {
                            if (result) {
                                instance.total = result.total || 0;
                                if (renderFunction) renderFunction();
                            }
                        });
                    } else if (renderFunction) {
                        renderFunction();
                    }
                }
            },
            
            changePageSize: function() {
                const instance = window[instanceId];
                if (!instance) return;
                
                const select = document.getElementById(config.pageSizeId);
                if (select) {
                    instance.pageSize = parseInt(select.value);
                    instance.page = 1;
                    
                    if (serverSide && dataLoader) {
                        dataLoader(instance.page, instance.pageSize).then(result => {
                            if (result) {
                                instance.total = result.total || 0;
                                if (renderFunction) renderFunction();
                            }
                        });
                    } else if (renderFunction) {
                        renderFunction();
                    }
                }
            }
        };
    }
};

