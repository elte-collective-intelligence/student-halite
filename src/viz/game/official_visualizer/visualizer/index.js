$(function () {
    var $dropZone = $("html");
    var $filePicker = $("#filePicker");

    function handleFiles(files) {
        let file = files[0];
        console.log(file);

        var reader = new FileReader();
        reader.onload = function (e) {
            $("#displayArea").empty();
            $("label[for=filePicker]").text("Select another file");
            var fsHeight = $("#fileSelect").outerHeight();
            showGame(textToGame(e.target.result, file.name), $("#displayArea"), null, -fsHeight, true, false, true);
        };
        reader.readAsText(file);
    }

    $dropZone.on('dragover', function (e) {
        e.preventDefault();
        e.stopPropagation();
    });

    $dropZone.on('drop', function (e) {
        e.preventDefault();
        e.stopPropagation();
        var files = e.originalEvent.dataTransfer.files;
        handleFiles(files);
    });

    $filePicker.on('change', function (e) {
        var files = e.target.files;
        handleFiles(files);
    });
});