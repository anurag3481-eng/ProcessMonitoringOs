# Enhanced theme configuration with multiple theme options
THEMES = {
    "dark": {
        "bg": "#1e1e2e",
        "card_bg": "#252526",
        "text": "#FFFFFF",
        "accent": "#007ACC",
        "accent_hover": "#5e73bc",
        "danger": "#f04747",
        "success": "#43b581",
        "warning": "#faa61a",
        "chart_bg": "#2a2a3c",
        "grid_color": "#444455",
        "cpu_color": "#7289da",
        "mem_color": "#43b581",
        "disk_color": "#faa61a",
        "name": "Dark Mode",
        "icon": "🌙",
        "button_bg": "#333333"
    },
    "light": {
        "bg": "#f5f7fa",
        "card_bg": "#FFFFFF",
        "text": "#000000",
        "accent": "#0078D4",
        "accent_hover": "#4752c4",
        "danger": "#ed4245",
        "success": "#3ba55c",
        "warning": "#faa81a",
        "chart_bg": "#ffffff",
        "grid_color": "#e3e5e8",
        "cpu_color": "#5865f2",
        "mem_color": "#3ba55c",
        "disk_color": "#faa81a",
        "name": "Light Mode",
        "icon": "☀️",
        "button_bg": "#e0e0e0"
    },
    "blue-tech": {
        "bg": "#0a192f",
        "card_bg": "#112240",
        "text": "#e6f1ff",
        "accent": "#64ffda",
        "accent_hover": "#00e5b9",
        "danger": "#ff6b6b",
        "success": "#00b894",
        "warning": "#fdcb6e",
        "chart_bg": "#112240",
        "grid_color": "#1d4073",
        "cpu_color": "#64ffda",
        "mem_color": "#00b894",
        "disk_color": "#fdcb6e",
        "name": "Blue Tech",
        "icon": "🌊"
    },
    "green-industrial": {
        "bg": "#222d32",
        "card_bg": "#2c3b41",
        "text": "#b8c7ce",
        "accent": "#00a65a",
        "accent_hover": "#008d4c",
        "danger": "#dd4b39",
        "success": "#00a65a",
        "warning": "#f39c12",
        "chart_bg": "#2c3b41",
        "grid_color": "#3c4b51",
        "cpu_color": "#00a65a",
        "mem_color": "#00c0ef",
        "disk_color": "#f39c12",
        "name": "Green Industrial",
        "icon": "🌿"
    },
    "night-purple": {
        "bg": "#1a1b27",
        "card_bg": "#252640",
        "text": "#c5cde6",
        "accent": "#bb9af7",
        "accent_hover": "#a384de",
        "danger": "#f7768e",
        "success": "#9ece6a",
        "warning": "#e0af68",
        "chart_bg": "#252640",
        "grid_color": "#3a3c60",
        "cpu_color": "#bb9af7",
        "mem_color": "#9ece6a",
        "disk_color": "#e0af68",
        "name": "Night Purple",
        "icon": "🌃"
    },
    "sunrise": {
        "bg": "#ffeadb",
        "card_bg": "#fff1e6",
        "text": "#592c20",
        "accent": "#ff7b73",
        "accent_hover": "#e56a63",
        "danger": "#e56a63",
        "success": "#91c788",
        "warning": "#ffac60",
        "chart_bg": "#fff1e6",
        "grid_color": "#ffe0cc",
        "cpu_color": "#ff7b73",
        "mem_color": "#91c788",
        "disk_color": "#ffac60",
        "name": "Sunrise",
        "icon": "🌅"
    }
}

# Add custom styles
CUSTOM_STYLES = {
    "Card.TFrame": {
        "configure": {
            "background": "{card_bg}",
            "relief": "flat",
            "borderwidth": 0
        }
    },
    "Search.TFrame": {
        "background": "card_bg",
        "borderwidth": 1,
        "relief": "solid",
        "bordercolor": "accent"
    },
    "Title.TLabel": {
        "font": ("Segoe UI", 14, "bold"),
        "foreground": "accent",
        "background": "card_bg",
        "padding": (5, 5)
    },
    "Info.TLabel": {
        "font": ("Segoe UI", 10),
        "foreground": "text",
        "background": "card_bg"
    },
    "Small.TButton": {
        "font": ("Segoe UI", 9),
        "padding": (5, 2)
    },
    "Custom.Treeview": {
        "rowheight": 25,
        "font": ("Segoe UI", 10),
        "background": "card_bg",
        "foreground": "text",
        "fieldbackground": "card_bg",
        "borderwidth": 0
    },
    "Custom.Treeview.Heading": {
        "font": ("Segoe UI", 10, "bold"),
        "background": "accent",
        "foreground": "text",
        "borderwidth": 0
    },
    "IconButton.TButton": {
        "configure": {
            "background": "{card_bg}",
            "foreground": "{text}",
            "borderwidth": 0,
            "focusthickness": 0,
            "font": ("Segoe UI", 11),
            "padding": 5
        },
        "map": {
            "background": [("active", "{accent_hover}")],
            "foreground": [("active", "{bg}")]
        }
    },
    "RoundedButton.TButton": {
        "configure": {
            "background": "{accent}",
            "foreground": "{bg}",
            "borderwidth": 0,
            "focusthickness": 0,
            "padding": (10, 5),
            "font": ("Segoe UI", 9, "bold")
        },
        "map": {
            "background": [("active", "{accent_hover}")],
            "foreground": [("active", "{bg}")]
        }
    },
    "AI.TButton": {
        "configure": {
            "background": "#252640",  # Dark background for AI section
            "foreground": "#c5cde6",  # Light text for contrast
            "borderwidth": 1,
            "focusthickness": 0,
            "padding": (8, 4),
            "font": ("Segoe UI", 9, "bold")
        },
        "map": {
            "background": [("active", "#3a3c60")],  # Slightly lighter when active
            "foreground": [("active", "#ffffff")]   # White text when active
        }
    },
    "AICard.TFrame": {
        "configure": {
            "background": "#252640",  # Dark background for AI section
            "relief": "flat",
            "borderwidth": 0
        }
    }
}

# Default alert thresholds
DEFAULT_ALERT_THRESHOLDS = {
    "cpu": 80,
    "memory": 80,
    "disk": 90
}

# Default refresh rate in seconds
DEFAULT_REFRESH_RATE = 1 
