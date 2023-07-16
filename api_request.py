import requests

import secrets_1

def hello_world():
    print("Hello, world!")
    return None

def make_rest_api_call(url, method='GET', data=None):
  """Makes a REST API call.

  Args:
    url: The URL of the REST API.
    method: The HTTP method to use.
    data: The data to send with the request.

  Returns:
    The response from the REST API.
  """

  response = requests.request(method, url, data=data)
#   print(response.status_code)
#   print('check')

  if response.status_code == 200:
    # The request was successful.
    response = response.json()
    print(response)
    return response
  else:
    # The request failed.
    print(response.status_code)
    return None


if __name__ == '__main__':
   hello_world()
   make_rest_api_call(url='https://www.eventbriteapi.com/v3/users/me/?token=' + secrets_1.eventbrite_token)
