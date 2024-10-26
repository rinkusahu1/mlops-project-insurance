import base64
import io
import matplotlib.pyplot as plt


# render_type can be 'jpeg' or 'jpg'
@render(render_type='jpeg')
def r(*args, **kwargs):
    # creating the dataset
    data = {'C':20, 'C++':15, 'Java':30,
            'Python':35}
    courses = list(data.keys())
    values = list(data.values())

    fig = plt.figure(figsize = (10, 5))

    # creating the bar plot
    plt.bar(courses, values, color ='maroon',
            width = 0.4)

    plt.xlabel("Courses offered")
    plt.ylabel("No. of students enrolled")
    plt.title("Students enrolled in different courses")
    plt.show()

    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()

    plt.close()

    return my_base64_jpgData
