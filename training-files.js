document.addEventListener('DOMContentLoaded', () => {
    const fileListContainer = document.getElementById('knowledge-file-list');
    const noFilesMessage = document.getElementById('no-files-message');

    async function loadFiles() {
        if (!fileListContainer) return;

        try {
            const response = await fetch('/api/train/files');
            if (!response.ok) {
                throw new Error('Failed to fetch files.');
            }
            const files = await response.json();

            // Clear previous content, but preserve the 'no files' message element
            while (fileListContainer.firstChild && fileListContainer.firstChild !== noFilesMessage) {
                fileListContainer.removeChild(fileListContainer.firstChild);
            }

            if (files.length === 0) {
                noFilesMessage.style.display = 'block';
            } else {
                noFilesMessage.style.display = 'none';
                files.forEach(filename => {
                    const fileItem = createFileListItem(filename);
                    fileListContainer.appendChild(fileItem);
                });
            }
        } catch (error) {
            console.error('Error loading files:', error);
            fileListContainer.innerHTML = '<p style="color: red;">Error loading files. Please refresh the page.</p>';
        }
    }

    function createFileListItem(filename) {
        const item = document.createElement('div');
        item.className = 'knowledge-file-item';
        item.dataset.filename = filename;

        const nameSpan = document.createElement('span');
        nameSpan.textContent = filename;

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'delete-file-btn';
        deleteBtn.title = `Delete ${filename}`;
        deleteBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
            </svg>
        `;

        deleteBtn.addEventListener('click', () => handleDeleteClick(filename));

        item.appendChild(nameSpan);
        item.appendChild(deleteBtn);
        return item;
    }

    async function handleDeleteClick(filename) {
        if (!confirm(`Are you sure you want to permanently delete the file '${filename}'? This cannot be undone.`)) {
            return;
        }

        try {
            const response = await fetch(`/api/train/files/${filename}`, { method: 'DELETE' });

            if (response.status === 401) {
                showToast('error', 'Your session has expired. Please log in again.');
                window.location.href = '/login';
                return;
            }

            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || 'Failed to delete file.');

            showToast('success', result.message);
            loadFiles(); // Reload the list to reflect the change

        } catch (error) {
            console.error('Error deleting file:', error);
            showToast('error', `Error: ${error.message}`);
        }
    