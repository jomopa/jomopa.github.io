console.log("Hello, world!");
document.querySelector('input[type="submit"]').addEventListener('click', function() {

    fetch('/execute', {method: 'POST'});

});
