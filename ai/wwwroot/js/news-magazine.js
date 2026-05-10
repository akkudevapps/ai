// dark/light mode with localStorage
(function () {
    const toggle = document.getElementById('darkModeToggle');
    const body = document.body;
    const storageKey = 'akku-theme';

    function setTheme(theme) {
        if (theme === 'dark') {
            body.classList.add('theme-dark');
            body.classList.remove('theme-auto');
        } else if (theme === 'light') {
            body.classList.remove('theme-dark');
            body.classList.remove('theme-auto');
        } else {
            // auto: follow system
            body.classList.remove('theme-dark');
            body.classList.add('theme-auto');
            const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            if (systemDark) {
                body.classList.add('theme-dark');
            } else {
                body.classList.remove('theme-dark');
            }
        }
        localStorage.setItem(storageKey, theme);
    }

    function getInitialTheme() {
        const saved = localStorage.getItem(storageKey);
        if (saved === 'dark' || saved === 'light') return saved;
        return 'auto';
    }

    setTheme(getInitialTheme());

    if (toggle) {
        toggle.addEventListener('click', () => {
            const current = localStorage.getItem(storageKey) || 'auto';
            if (current === 'dark') setTheme('light');
            else if (current === 'light') setTheme('dark');
            else setTheme('dark'); // from auto -> dark
        });
    }

    // listen to system changes if in auto mode
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        if (localStorage.getItem(storageKey) === 'auto') {
            if (e.matches) body.classList.add('theme-dark');
            else body.classList.remove('theme-dark');
        }
    });
})();