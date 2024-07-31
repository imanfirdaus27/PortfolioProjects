<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Slides</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@4/dist/reveal.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@4/dist/theme/white.css" id="theme">
    <link rel="stylesheet" href="custom.css">
    <script src="https://cdn.jsdelivr.net/npm/reveal.js@4/dist/reveal.js"></script>
    <style>
        table {
            width: 100%;
            overflow-x: auto;
            display: block;
            white-space: nowrap;
        }
        .reveal .slides {
            font-size: 18px; /* Adjust font size if needed */
        }
    </style>
</head>
<body>
    <div class="reveal">
        <div class="slides">
            {{ body|safe }}
        </div>
    </div>
    <script>
        Reveal.initialize();
    </script>
</body>
</html>
    