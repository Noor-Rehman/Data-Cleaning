// assets/clientside-error-handler.js
// This script handles chunk/component loading errors in Dash and displays a user-friendly message instead of breaking the app.

(function() {
    if (window.__dash_chunk_load_error_handler_installed) return;
    window.__dash_chunk_load_error_handler_installed = true;

    window.addEventListener('error', function(event) {
        if (event && event.message && event.message.indexOf('Loading chunk') !== -1) {
            var errorDiv = document.createElement('div');
            errorDiv.style.position = 'fixed';
            errorDiv.style.top = '0';
            errorDiv.style.left = '0';
            errorDiv.style.width = '100vw';
            errorDiv.style.background = '#ff1744';
            errorDiv.style.color = '#fff';
            errorDiv.style.fontSize = '1.2rem';
            errorDiv.style.fontWeight = 'bold';
            errorDiv.style.zIndex = '9999';
            errorDiv.style.padding = '1rem 2rem';
            errorDiv.style.textAlign = 'center';
            errorDiv.innerText = 'A required component failed to load. Please refresh the page. If the problem persists, clear your browser cache.';
            document.body.appendChild(errorDiv);
        }
    }, true);
})();window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        handleChunkError: function(children) {
            if (window._dashChunkErrorHandled) return window.dash_clientside.no_update;
            window._dashChunkErrorHandled = true;
            if (window && window.document) {
                var errorDiv = document.createElement('div');
                errorDiv.style.background = '#ff1744';
                errorDiv.style.color = '#fff';
                errorDiv.style.fontWeight = 'bold';
                errorDiv.style.padding = '1rem';
                errorDiv.style.margin = '1rem 0';
                errorDiv.innerText = 'A component failed to load. Please refresh the page. If the problem persists, clear your browser cache.';
                var container = document.getElementById('alert-container');
                if (container) container.appendChild(errorDiv);
            }
            return window.dash_clientside.no_update;
        }
    }
});
