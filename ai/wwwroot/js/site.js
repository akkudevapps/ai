// Please see documentation at https://learn.microsoft.com/aspnet/core/client-side/bundling-and-minification
// for details on configuring this project to bundle and minify static web assets.

// Write your JavaScript code.
// site.js (உங்கள் ASP.NET project-ல் சேர்க்க)
async function askAkkuChatbot(message) {
    try {
        let response = await fetch('http://localhost:5000/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        let data = await response.json();
        return data.reply;
    } catch (error) {
        console.error('Error:', error);
        return 'மன்னிக்கவும், சேவையகத்தில் பிழை ஏற்பட்டுள்ளது.';
    }
}

// Floating Chat Toggle
document.getElementById('chatToggle').addEventListener('click', () => {
    document.getElementById('chatWidget').classList.toggle('hidden');
});