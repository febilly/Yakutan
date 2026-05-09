function setupAutoExpandCollapsibleOnFocus() {
    document.querySelectorAll('.collapsible-content').forEach(content => {
        content.addEventListener('focusin', (event) => {
            if (content.classList.contains('collapsed') && content.id) {
                toggleCollapsible(content.id);
            }
        });
    });
}
